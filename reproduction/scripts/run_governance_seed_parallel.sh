#!/usr/bin/env bash
set -euo pipefail

if [[ "${CONFIRM_SHARED_GPU:-0}" != "1" ]]; then
  echo "Refusing to start: set CONFIRM_SHARED_GPU=1 after accepting shared-GPU safeguards." >&2
  exit 2
fi
if [[ -z "${SEED:-}" ]]; then
  echo "SEED is required." >&2
  exit 2
fi

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/nfs/bkbs/projects/ZJU-Graduation-Project/reproduction}"
CONDA_INIT="${CONDA_INIT:-/home/bkbs/miniforge3/etc/profile.d/conda.sh}"
CONFIG="${CONFIG:-configs/governance_formal.yaml}"
GPU_INDEX="${GPU_INDEX:-0}"
MAX_INITIAL_MEMORY_MIB="${MAX_INITIAL_MEMORY_MIB:-1024}"
MAX_INITIAL_UTILIZATION="${MAX_INITIAL_UTILIZATION:-10}"
GPU_RECHECK_SECONDS="${GPU_RECHECK_SECONDS:-60}"

source "$CONDA_INIT"
conda activate tabular-benchmark
cd "$PROJECT_ROOT"
(cd configs && sha256sum -c governance_formal.sha256)

OUTPUT_DIR="$(python -c 'from tabpollution.governance.config import load_governance_config; import sys; print(load_governance_config(sys.argv[1]).output_dir)' "$CONFIG")"
mkdir -p "$OUTPUT_DIR/scheduler"
exec 9>"$OUTPUT_DIR/scheduler/formal_scheduler.lock"
if ! flock -n 9; then
  echo "Another formal scheduler already holds $OUTPUT_DIR/scheduler/formal_scheduler.lock" >&2
  exit 3
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

python -m tabpollution environment capture --output "$OUTPUT_DIR/scheduler/environment-seed-${SEED}.txt"
python -m tabpollution governance preflight --config "$CONFIG"

queue() {
  local resource_class="$1"
  python -c 'from tabpollution.governance.shards import shard_queue; import sys; sys.stdout.write("\n".join(shard_queue(sys.argv[1], seed=int(sys.argv[2]), resource_class=sys.argv[3])["pending_shards"]))' "$CONFIG" "$SEED" "$resource_class"
}

mapfile -t CPU_SHARDS < <(queue cpu)
mapfile -t GPU_SHARDS < <(queue gpu)
echo "seed=$SEED cpu_pending=${#CPU_SHARDS[@]} gpu_pending=${#GPU_SHARDS[@]}"

run_cpu_queue() {
  local shard
  export CUDA_VISIBLE_DEVICES=""
  for shard in "${CPU_SHARDS[@]}"; do
    [[ -n "$shard" ]] || continue
    echo "CPU_START $shard $(date --iso-8601=seconds)"
    nice -n 15 python -m tabpollution governance shard-run \
      --config "$CONFIG" --shard-id "$shard" --resume
    echo "CPU_DONE $shard $(date --iso-8601=seconds)"
  done
}

gpu_is_idle() {
  local memory_used utilization apps
  IFS=, read -r memory_used utilization < <(
    nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits | tr -d ' '
  )
  apps="$(nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')"
  [[ -z "$apps" && "$memory_used" -le "$MAX_INITIAL_MEMORY_MIB" && "$utilization" -le "$MAX_INITIAL_UTILIZATION" ]]
}

wait_for_gpu() {
  until gpu_is_idle; do
    echo "GPU_BUSY waiting ${GPU_RECHECK_SECONDS}s $(date --iso-8601=seconds)"
    sleep "$GPU_RECHECK_SECONDS"
  done
}

run_gpu_shard() {
  local shard="$1" observed_pid run_status external_seen
  while true; do
    wait_for_gpu
    external_seen=0
    export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
    echo "GPU_START $shard $(date --iso-8601=seconds)"
    nice -n 10 python -m tabpollution governance shard-run \
      --config "$CONFIG" --shard-id "$shard" --resume &
    GPU_RUN_PID=$!
    while kill -0 "$GPU_RUN_PID" 2>/dev/null; do
      while IFS= read -r observed_pid; do
        observed_pid="${observed_pid//[[:space:]]/}"
        if [[ -n "$observed_pid" && "$observed_pid" != "$GPU_RUN_PID" ]]; then
          echo "EXTERNAL_GPU_PROCESS pid=$observed_pid; yielding shared GPU" >&2
          external_seen=1
          kill -TERM "$GPU_RUN_PID" 2>/dev/null || true
          break
        fi
      done < <(nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid --format=csv,noheader,nounits)
      [[ "$external_seen" -eq 0 ]] || break
      sleep 10
    done
    set +e
    wait "$GPU_RUN_PID"
    run_status=$?
    set -e
    GPU_RUN_PID=""
    if [[ "$run_status" -eq 0 ]]; then
      echo "GPU_DONE $shard $(date --iso-8601=seconds)"
      return 0
    fi
    if [[ "$external_seen" -eq 1 ]]; then
      echo "GPU_RETRY_AFTER_YIELD $shard"
      sleep "$GPU_RECHECK_SECONDS"
      continue
    fi
    echo "GPU_FAILED $shard status=$run_status" >&2
    return "$run_status"
  done
}

run_gpu_queue() {
  local shard
  for shard in "${GPU_SHARDS[@]}"; do
    [[ -n "$shard" ]] || continue
    run_gpu_shard "$shard"
  done
}

run_cpu_queue >"$OUTPUT_DIR/scheduler/seed-${SEED}-cpu.log" 2>&1 &
cpu_pid=$!
run_gpu_queue >"$OUTPUT_DIR/scheduler/seed-${SEED}-gpu.log" 2>&1 &
gpu_pid=$!
cleanup_workers() {
  kill -TERM "$cpu_pid" "$gpu_pid" 2>/dev/null || true
}
trap cleanup_workers INT TERM

set +e
wait "$cpu_pid"
cpu_status=$?
wait "$gpu_pid"
gpu_status=$?
set -e
trap - INT TERM

python -m tabpollution governance shard-status --config "$CONFIG" >"$OUTPUT_DIR/scheduler/seed-${SEED}-status.json"
if [[ "$cpu_status" -ne 0 || "$gpu_status" -ne 0 ]]; then
  echo "Scheduler failed: cpu_status=$cpu_status gpu_status=$gpu_status" >&2
  exit 5
fi
echo "SEED_DONE $SEED $(date --iso-8601=seconds)"
