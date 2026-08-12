#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${SEED:-}" ]]; then
  echo "SEED is required." >&2
  exit 2
fi
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/nfs/bkbs/projects/ZJU-Graduation-Project/reproduction}"
CONDA_INIT="${CONDA_INIT:-/home/bkbs/miniforge3/etc/profile.d/conda.sh}"
CONFIG="${CONFIG:-configs/governance_formal.yaml}"

source "$CONDA_INIT"
conda activate tabular-benchmark
cd "$PROJECT_ROOT"
(cd configs && sha256sum -c governance_formal.sha256)
OUTPUT_DIR="$(python -c 'from tabpollution.governance.config import load_governance_config; import sys; print(load_governance_config(sys.argv[1]).output_dir)' "$CONFIG")"
mkdir -p "$OUTPUT_DIR/scheduler"
exec 8>"$OUTPUT_DIR/scheduler/cpu-worker.lock"
if ! flock -n 8; then
  echo "Another formal CPU worker is active." >&2
  exit 3
fi

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

mapfile -t SHARDS < <(
  python -c 'from tabpollution.governance.shards import shard_queue; import sys; sys.stdout.write("\n".join(shard_queue(sys.argv[1], seed=int(sys.argv[2]), resource_class="cpu")["pending_shards"]))' "$CONFIG" "$SEED"
)
echo "seed=$SEED cpu_pending=${#SHARDS[@]}"
for shard in "${SHARDS[@]}"; do
  [[ -n "$shard" ]] || continue
  echo "CPU_START $shard $(date --iso-8601=seconds)"
  nice -n 15 python -m tabpollution governance shard-run \
    --config "$CONFIG" --shard-id "$shard" --resume --execution-device cpu
  echo "CPU_DONE $shard $(date --iso-8601=seconds)"
done
echo "CPU_SEED_DONE $SEED $(date --iso-8601=seconds)"
