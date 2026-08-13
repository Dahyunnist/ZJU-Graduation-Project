#!/usr/bin/env bash
set -euo pipefail

if [[ "${CONFIRM_SHARED_GPU:-0}" != "1" ]]; then
  echo "CONFIRM_SHARED_GPU=1 is required." >&2
  exit 2
fi
if [[ -z "${WAIT_SEED:-}" || -z "${NEXT_SEED:-}" ]]; then
  echo "WAIT_SEED and NEXT_SEED are required." >&2
  exit 2
fi

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/nfs/bkbs/projects/ZJU-Graduation-Project/reproduction}"
CONDA_INIT="${CONDA_INIT:-/home/bkbs/miniforge3/etc/profile.d/conda.sh}"
CONFIG="${CONFIG:-configs/governance_formal.yaml}"
POLL_SECONDS="${POLL_SECONDS:-60}"

source "$CONDA_INIT"
conda activate tabular-benchmark
cd "$PROJECT_ROOT"
(cd configs && sha256sum -c governance_formal.sha256)

OUTPUT_DIR="$(python -c 'from tabpollution.governance.config import load_governance_config; import sys; print(load_governance_config(sys.argv[1]).output_dir)' "$CONFIG")"
LOCK_PATH="$OUTPUT_DIR/scheduler/formal_scheduler.lock"
mkdir -p "$OUTPUT_DIR/scheduler"
exec 7>"$LOCK_PATH"

echo "RELAY_WAIT wait_seed=$WAIT_SEED next_seed=$NEXT_SEED $(date --iso-8601=seconds)"
while ! flock -n 7; do
  sleep "$POLL_SECONDS"
done
flock -u 7

pending() {
  local seed="$1" resource_class="$2"
  python -c 'from tabpollution.governance.shards import shard_queue; import sys; print(shard_queue(sys.argv[1], seed=int(sys.argv[2]), resource_class=sys.argv[3])["pending_count"])' "$CONFIG" "$seed" "$resource_class"
}

cpu_pending="$(pending "$WAIT_SEED" cpu)"
gpu_pending="$(pending "$WAIT_SEED" gpu)"
echo "RELAY_CHECK seed=$WAIT_SEED cpu_pending=$cpu_pending gpu_pending=$gpu_pending $(date --iso-8601=seconds)"
if [[ "$cpu_pending" -ne 0 || "$gpu_pending" -ne 0 ]]; then
  echo "RELAY_ABORT seed $WAIT_SEED did not complete; refusing to start $NEXT_SEED" >&2
  exit 4
fi

echo "RELAY_START seed=$NEXT_SEED $(date --iso-8601=seconds)"
exec env CONFIRM_SHARED_GPU=1 SEED="$NEXT_SEED" \
  bash scripts/run_governance_seed_parallel.sh
