#!/usr/bin/env bash
set -euo pipefail

if [[ "${CONFIRM_SHARED_GPU:-0}" != "1" ]]; then
  echo "Refusing to train generators: inspect nvidia-smi, then set CONFIRM_SHARED_GPU=1." >&2
  exit 2
fi

PROJECT_ROOT="/mnt/nfs/bkbs/projects/ZJU-Graduation-Project/reproduction"
CONDA_INIT="/home/bkbs/miniforge3/etc/profile.d/conda.sh"
GPU_INDEX="${GPU_INDEX:-0}"
MAX_INITIAL_MEMORY_MIB="${MAX_INITIAL_MEMORY_MIB:-1024}"
MAX_INITIAL_UTILIZATION="${MAX_INITIAL_UTILIZATION:-10}"

source "$CONDA_INIT"
conda activate tabular-benchmark
cd "$PROJECT_ROOT"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=-1

python -m tabpollution governance source-prepare --config configs/governance_pool_build.yaml
mkdir -p data/governance
python -m tabpollution environment capture --output data/governance/environment_pool_build.txt
python -m tabpollution governance pool-preflight --config configs/governance_pool_build.yaml

IFS=, read -r memory_used utilization < <(
  nvidia-smi -i "$GPU_INDEX" \
    --query-gpu=memory.used,utilization.gpu \
    --format=csv,noheader,nounits | tr -d ' '
)
if (( memory_used > MAX_INITIAL_MEMORY_MIB || utilization > MAX_INITIAL_UTILIZATION )); then
  echo "Refusing formal pool build: GPU $GPU_INDEX is not idle (memory=${memory_used}MiB, utilization=${utilization}%)." >&2
  exit 4
fi
if [[ -n "$(nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid --format=csv,noheader,nounits)" ]]; then
  echo "Refusing formal pool build: GPU $GPU_INDEX already has a compute process." >&2
  exit 4
fi

export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
nice -n 10 python -m tabpollution governance pool-build \
  --config configs/governance_pool_build.yaml --resume &
build_pid=$!
terminate_build() {
  if kill -0 "$build_pid" 2>/dev/null; then
    kill -TERM "$build_pid" 2>/dev/null || true
    wait "$build_pid" || true
  fi
}
trap terminate_build INT TERM

# Protect the shared card: if any compute PID other than this build appears,
# terminate only this user's resumable build.
while kill -0 "$build_pid" 2>/dev/null; do
  while IFS= read -r observed_pid; do
    observed_pid="${observed_pid//[[:space:]]/}"
    if [[ -n "$observed_pid" && "$observed_pid" != "$build_pid" ]]; then
      echo "Another GPU process ($observed_pid) appeared; terminating formal pool build $build_pid." >&2
      terminate_build
      exit 5
    fi
  done < <(nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid --format=csv,noheader,nounits)
  sleep 10
done
wait "$build_pid"
