#!/usr/bin/env bash
set -euo pipefail

if [[ "${CONFIRM_SHARED_GPU:-0}" != "1" ]]; then
  echo "Refusing pilot: inspect the shared GPU, then set CONFIRM_SHARED_GPU=1." >&2
  exit 2
fi

PROJECT_ROOT="/mnt/nfs/bkbs/projects/ZJU-Graduation-Project/reproduction"
PYTHON="/home/bkbs/miniforge3/envs/tabular-benchmark/bin/python"
GPU_INDEX="${GPU_INDEX:-0}"
MAX_INITIAL_MEMORY_MIB="${MAX_INITIAL_MEMORY_MIB:-1024}"
MAX_INITIAL_UTILIZATION="${MAX_INITIAL_UTILIZATION:-10}"

if [[ ! -d "$PROJECT_ROOT" || ! -x "$PYTHON" ]]; then
  echo "Expected project or personal Python environment is missing." >&2
  exit 3
fi

IFS=, read -r memory_used utilization < <(
  nvidia-smi -i "$GPU_INDEX" \
    --query-gpu=memory.used,utilization.gpu \
    --format=csv,noheader,nounits | tr -d ' '
)
if (( memory_used > MAX_INITIAL_MEMORY_MIB || utilization > MAX_INITIAL_UTILIZATION )); then
  echo "Refusing pilot: GPU $GPU_INDEX is not idle (memory=${memory_used}MiB, utilization=${utilization}%)." >&2
  exit 4
fi
if [[ -n "$(nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid --format=csv,noheader,nounits)" ]]; then
  echo "Refusing pilot: GPU $GPU_INDEX already has a compute process." >&2
  exit 4
fi

cd "$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

mkdir -p runs/governance-pool-pilot-v1
"$PYTHON" -m tabpollution environment capture \
  --output runs/governance-pool-pilot-v1/environment.txt
"$PYTHON" -m tabpollution governance pool-preflight \
  --config configs/governance_pool_build_pilot.yaml

nice -n 10 "$PYTHON" -m tabpollution governance pool-build \
  --config configs/governance_pool_build_pilot.yaml --resume &
pilot_pid=$!
watchdog_pid=""
cleanup_pilot() {
  if kill -0 "$pilot_pid" 2>/dev/null; then
    kill -TERM "$pilot_pid" 2>/dev/null || true
  fi
  if [[ -n "$watchdog_pid" ]]; then
    kill "$watchdog_pid" 2>/dev/null || true
  fi
}
trap cleanup_pilot INT TERM

# If another compute process appears, stop only this user's pilot.
(
  while kill -0 "$pilot_pid" 2>/dev/null; do
    while IFS= read -r observed_pid; do
      observed_pid="${observed_pid//[[:space:]]/}"
      if [[ -n "$observed_pid" && "$observed_pid" != "$pilot_pid" ]]; then
        echo "Another GPU process ($observed_pid) appeared; terminating pilot $pilot_pid." >&2
        kill -TERM "$pilot_pid" 2>/dev/null || true
        exit 5
      fi
    done < <(nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid --format=csv,noheader,nounits)
    sleep 10
  done
) &
watchdog_pid=$!

set +e
wait "$pilot_pid"
pilot_status=$?
set -e
kill "$watchdog_pid" 2>/dev/null || true
wait "$watchdog_pid" 2>/dev/null || true
trap - INT TERM
exit "$pilot_status"
