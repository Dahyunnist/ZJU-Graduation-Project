#!/usr/bin/env bash
# Explicit, resumable shared-GPU entry point. Do not call while a card is busy.
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 REPOSITORY BASE_POOL OUTPUT_POOL MODEL_ROOT GPU_INDEX" >&2
  exit 2
fi
if [[ "${CONFIRM_SHARED_GPU:-0}" != "1" ]]; then
  echo "Refusing E3 training without CONFIRM_SHARED_GPU=1" >&2
  exit 2
fi
repo=$1
base=$2
output=$3
models=$4
gpu=$5
base="$(realpath -m "$base")"
output="$(realpath -m "$output")"
models="$(realpath -m "$models")"
if [[ "$base" != "/mnt/nfs/bkbs/datasets/ZJU-Graduation-Project/governance" ||
      "$output" != /mnt/nfs/bkbs/datasets/ZJU-Graduation-Project/governance-e3-* ||
      "$models" != /mnt/nfs/bkbs/checkpoints/ZJU-Graduation-Project/e3-* ||
      ! "$gpu" =~ ^[0-9]+$ ]]; then
  echo "E3 path/GPU boundary check failed; no training." >&2
  exit 2
fi
python=/home/bkbs/miniforge3/envs/tabular-benchmark/bin/python
config="$repo/reproduction/configs/e3_independent_pools_v1.yaml"

mkdir -p "$output"
exec 9>"$output/worker.lock"
if ! flock -n 9; then
  echo "E3 pool worker already active" >&2
  exit 3
fi
record_status() {
  printf '%s\t%s\n' "$(date -u +%FT%TZ)" "$1" > "$output/job-status.txt.partial"
  mv "$output/job-status.txt.partial" "$output/job-status.txt"
}
export PYTHONPATH="$repo/reproduction/src"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
"$python" "$repo/reproduction/scripts/run_e3_pools.py" --config "$config" \
  --base-root "$base" --output-root "$output" --checkpoint-root "$models" \
  > "$output/preflight.json"

IFS=, read -r memory_used utilization < <(
  nvidia-smi -i "$gpu" --query-gpu=memory.used,utilization.gpu \
    --format=csv,noheader,nounits | tr -d ' '
)
if (( memory_used > 1024 || utilization > 10 )); then
  echo "Shared GPU $gpu is busy (${memory_used} MiB, ${utilization}%); no E3 training." >&2
  exit 4
fi
if [[ -n "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)" ]]; then
  echo "Shared GPU $gpu has another compute process; no E3 training." >&2
  exit 4
fi

export CUDA_VISIBLE_DEVICES="$gpu"
export E3_GPU_TRAINING_APPROVED=1
record_status "running pid=$$"
nice -n 10 "$python" "$repo/reproduction/scripts/run_e3_pools.py" \
  --config "$config" --base-root "$base" --output-root "$output" \
  --checkpoint-root "$models" --run --resume >> "$output/worker.log" 2>&1 &
build_pid=$!
watchdog_pid=""
cleanup() {
  if kill -0 "$build_pid" 2>/dev/null; then
    kill -TERM "$build_pid" 2>/dev/null || true
  fi
  if [[ -n "$watchdog_pid" ]]; then
    kill "$watchdog_pid" 2>/dev/null || true
  fi
}
trap cleanup INT TERM
(
  while kill -0 "$build_pid" 2>/dev/null; do
    while IFS= read -r observed_pid; do
      observed_pid="${observed_pid//[[:space:]]/}"
      if [[ -n "$observed_pid" && "$observed_pid" != "$build_pid" ]]; then
        echo "Another compute PID appeared; stopping only E3 build $build_pid" >&2
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
status=$?
set -e
kill "$watchdog_pid" 2>/dev/null || true
wait "$watchdog_pid" 2>/dev/null || true
trap - INT TERM
if (( status == 0 )); then record_status "complete"; else record_status "failed exit=$status"; fi
exit "$status"
