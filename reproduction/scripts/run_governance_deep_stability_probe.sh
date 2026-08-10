#!/usr/bin/env bash
set -euo pipefail

if [[ "${CONFIRM_SHARED_GPU:-0}" != "1" ]]; then
  echo "Refusing stability probe: inspect the shared GPU, then set CONFIRM_SHARED_GPU=1." >&2
  exit 2
fi

PROJECT_ROOT="/mnt/nfs/bkbs/projects/ZJU-Graduation-Project/reproduction"
CONDA_INIT="/home/bkbs/miniforge3/etc/profile.d/conda.sh"
GPU_INDEX="${GPU_INDEX:-0}"
MAX_INITIAL_MEMORY_MIB="${MAX_INITIAL_MEMORY_MIB:-1024}"
MAX_INITIAL_UTILIZATION="${MAX_INITIAL_UTILIZATION:-10}"

if [[ ! -d "$PROJECT_ROOT" || ! -f "$CONDA_INIT" ]]; then
  echo "Expected project or Conda initialization file is missing." >&2
  exit 3
fi

source "$CONDA_INIT"
conda activate tabular-benchmark
cd "$PROJECT_ROOT"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4

IFS=, read -r memory_used utilization < <(
  nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.used,utilization.gpu \
    --format=csv,noheader,nounits | tr -d ' '
)
if (( memory_used > MAX_INITIAL_MEMORY_MIB || utilization > MAX_INITIAL_UTILIZATION )); then
  echo "Refusing stability probe: GPU $GPU_INDEX is not idle (memory=${memory_used}MiB, utilization=${utilization}%)." >&2
  exit 4
fi
if [[ -n "$(nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid --format=csv,noheader,nounits)" ]]; then
  echo "Refusing stability probe: GPU $GPU_INDEX already has a compute process." >&2
  exit 4
fi

export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
OUTPUT_DIR="runs/governance-deep-stability-probe-v1"
mkdir -p "$OUTPUT_DIR"
python -m tabpollution environment capture --output "$OUTPUT_DIR/environment.txt"
python -m tabpollution governance preflight --config configs/governance_deep_stability_probe.yaml

nice -n 10 python -m tabpollution governance run \
  --config configs/governance_deep_stability_probe.yaml &
run_pid=$!
watchdog_pid=""
cleanup_run() {
  if kill -0 "$run_pid" 2>/dev/null; then kill -TERM "$run_pid" 2>/dev/null || true; fi
  if [[ -n "$watchdog_pid" ]]; then kill "$watchdog_pid" 2>/dev/null || true; fi
}
trap cleanup_run INT TERM

(
  while kill -0 "$run_pid" 2>/dev/null; do
    while IFS= read -r observed_pid; do
      observed_pid="${observed_pid//[[:space:]]/}"
      if [[ -n "$observed_pid" && "$observed_pid" != "$run_pid" ]]; then
        echo "Another GPU process ($observed_pid) appeared; terminating stability probe $run_pid." >&2
        kill -TERM "$run_pid" 2>/dev/null || true
        exit 5
      fi
    done < <(nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid --format=csv,noheader,nounits)
    sleep 10
  done
) &
watchdog_pid=$!

set +e
wait "$run_pid"
run_status=$?
set -e
kill "$watchdog_pid" 2>/dev/null || true
wait "$watchdog_pid" 2>/dev/null || true
trap - INT TERM
exit "$run_status"
