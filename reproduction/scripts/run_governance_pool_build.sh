#!/usr/bin/env bash
set -euo pipefail

if [[ "${CONFIRM_SHARED_GPU:-0}" != "1" ]]; then
  echo "Refusing to train generators: inspect nvidia-smi, then set CONFIRM_SHARED_GPU=1." >&2
  exit 2
fi

PROJECT_ROOT="/mnt/nfs/bkbs/projects/ZJU-Graduation-Project/reproduction"
CONDA_INIT="/home/bkbs/miniforge3/etc/profile.d/conda.sh"

source "$CONDA_INIT"
conda activate tabular-benchmark
cd "$PROJECT_ROOT"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

nvidia-smi
python -m tabpollution governance source-prepare --config configs/governance_pool_build.yaml
mkdir -p data/governance
python -m tabpollution environment capture --output data/governance/environment_pool_build.txt
python -m tabpollution governance pool-preflight --config configs/governance_pool_build.yaml
python -m tabpollution governance pool-build --config configs/governance_pool_build.yaml --resume
