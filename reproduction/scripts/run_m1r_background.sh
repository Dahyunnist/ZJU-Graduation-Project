#!/usr/bin/env bash
# Run only this project's frozen M1R job after the caller has gated one idle GPU.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 REPOSITORY GPU_INDEX" >&2
  exit 2
fi
if [[ "${CONFIRM_SHARED_GPU:-0}" != "1" ]]; then
  echo "Refusing M1R training without CONFIRM_SHARED_GPU=1" >&2
  exit 2
fi

repo=$(realpath -m "$1")
gpu=$2
python=/home/bkbs/miniforge3/envs/tabular-benchmark/bin/python
base=/mnt/nfs/bkbs/datasets/ZJU-Graduation-Project/governance
datasets=/mnt/nfs/bkbs/datasets/ZJU-Graduation-Project
models=/mnt/nfs/bkbs/checkpoints/ZJU-Graduation-Project
results=/mnt/nfs/bkbs/results/ZJU-Graduation-Project
control="$results/m1r-launch-v1"
config="$repo/reproduction/configs/m1r_independent_repeats_v1.yaml"

if [[ "$repo" != "/mnt/nfs/bkbs/projects/ZJU-Graduation-Project" || ! "$gpu" =~ ^[0-9]+$ ]]; then
  echo "M1R repository/GPU boundary check failed" >&2
  exit 2
fi
mkdir -p "$control"
exec 9>"$control/worker.lock"
if ! flock -n 9; then
  echo "Another M1R worker is active" >&2
  exit 3
fi

status() {
  printf '%s\t%s\n' "$(date -u +%FT%TZ)" "$1" > "$control/job-status.txt.partial"
  mv "$control/job-status.txt.partial" "$control/job-status.txt"
}

export PYTHONPATH="$repo/reproduction/src"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=""

status "preflight"
nice -n 10 "$python" "$repo/reproduction/scripts/run_m1r.py" \
  --config "$config" --repository-root "$repo" --base-root "$base" \
  --dataset-root "$datasets" --checkpoint-root "$models" --results-root "$results" \
  --phase preflight > "$control/preflight-before-training.json"

IFS=, read -r memory_used utilization < <(
  nvidia-smi -i "$gpu" --query-gpu=memory.used,utilization.gpu \
    --format=csv,noheader,nounits | tr -d ' '
)
if (( memory_used > 1024 || utilization > 10 )); then
  echo "Shared GPU $gpu became busy; no M1R training" >&2
  exit 4
fi
if [[ -n "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)" ]]; then
  echo "Shared GPU $gpu has another compute process; no M1R training" >&2
  exit 4
fi

export CUDA_VISIBLE_DEVICES="$gpu"
export M1R_GPU_TRAINING_APPROVED=1
status "training gpu=$gpu pid=$$"
nice -n 10 "$python" "$repo/reproduction/scripts/run_m1r.py" \
  --config "$config" --repository-root "$repo" --base-root "$base" \
  --dataset-root "$datasets" --checkpoint-root "$models" --results-root "$results" \
  --phase train >> "$control/worker.log" 2>&1 &
build_pid=$!
watchdog_pid=""
cleanup() {
  if kill -0 "$build_pid" 2>/dev/null; then kill -TERM "$build_pid" 2>/dev/null || true; fi
  if [[ -n "$watchdog_pid" ]]; then kill "$watchdog_pid" 2>/dev/null || true; fi
}
trap cleanup INT TERM
(
  while kill -0 "$build_pid" 2>/dev/null; do
    while IFS= read -r observed_pid; do
      observed_pid="${observed_pid//[[:space:]]/}"
      if [[ -n "$observed_pid" && "$observed_pid" != "$build_pid" ]]; then
        echo "Another compute PID appeared; stopping only M1R build $build_pid" >&2
        kill -TERM "$build_pid" 2>/dev/null || true
        exit 5
      fi
    done < <(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)
    sleep 10
  done
) &
watchdog_pid=$!
set +e
wait "$build_pid"
build_status=$?
set -e
kill "$watchdog_pid" 2>/dev/null || true
wait "$watchdog_pid" 2>/dev/null || true
trap - INT TERM
if (( build_status != 0 )); then
  status "failed stage=training exit=$build_status"
  exit "$build_status"
fi

export CUDA_VISIBLE_DEVICES=""
status "pool_audit"
nice -n 10 "$python" "$repo/reproduction/scripts/run_m1r.py" \
  --config "$config" --repository-root "$repo" --base-root "$base" \
  --dataset-root "$datasets" --checkpoint-root "$models" --results-root "$results" \
  --phase preflight > "$control/preflight-after-training.json"

status "detecting"
nice -n 10 "$python" "$repo/reproduction/scripts/run_m1r.py" \
  --config "$config" --repository-root "$repo" --base-root "$base" \
  --dataset-root "$datasets" --checkpoint-root "$models" --results-root "$results" \
  --phase detect >> "$control/worker.log" 2>&1

status "auditing"
audit="$results/m1r-independent-repeats-audit-v1"
if [[ ! -d "$audit" ]]; then
  nice -n 10 "$python" "$repo/reproduction/scripts/analyze_m1r.py" \
    --config "$config" --results-root "$results" --output "$audit" \
    >> "$control/worker.log" 2>&1
fi
status "complete"

