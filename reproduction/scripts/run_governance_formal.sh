#!/usr/bin/env bash
set -euo pipefail

if [[ "${CONFIRM_SHARED_GPU:-0}" != "1" ]]; then
  echo "Refusing to start: inspect nvidia-smi, then set CONFIRM_SHARED_GPU=1." >&2
  exit 2
fi

PROJECT_ROOT="/mnt/nfs/bkbs/projects/ZJU-Graduation-Project/reproduction"
CONDA_INIT="/home/bkbs/miniforge3/etc/profile.d/conda.sh"

if [[ ! -d "$PROJECT_ROOT" || ! -f "$CONDA_INIT" ]]; then
  echo "Expected project or Conda initialization file is missing." >&2
  exit 3
fi

source "$CONDA_INIT"
conda activate tabular-benchmark
cd "$PROJECT_ROOT"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader
python -m tabpollution governance preflight --config configs/governance_formal.yaml
python -m tabpollution governance run --config configs/governance_formal.yaml
