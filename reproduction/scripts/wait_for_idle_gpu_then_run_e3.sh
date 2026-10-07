#!/usr/bin/env bash
# Polls only GPU metadata; starts one E3 build after sustained per-card idleness.
# The build itself rechecks the card and stops *only itself* on contention.
set -euo pipefail

repo=/mnt/nfs/bkbs/projects/ZJU-Graduation-Project
base=/mnt/nfs/bkbs/datasets/ZJU-Graduation-Project/governance
output=/mnt/nfs/bkbs/datasets/ZJU-Graduation-Project/governance-e3-r3031
models=/mnt/nfs/bkbs/checkpoints/ZJU-Graduation-Project/e3-r3031
control=/mnt/nfs/bkbs/results/ZJU-Graduation-Project/e3-pool-auto-r3031
mkdir -p "$control"
exec 9>"$control/wait.lock"
if ! flock -n 9; then
  echo "Another E3 GPU waiter is already active" >&2
  exit 3
fi

record_status() {
  printf '%s\t%s\n' "$(date -u +%FT%TZ)" "$1" > "$control/wait-status.txt.partial"
  mv "$control/wait-status.txt.partial" "$control/wait-status.txt"
}
child_pid=""
on_signal() {
  if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
    kill -TERM "$child_pid" 2>/dev/null || true
    wait "$child_pid" 2>/dev/null || true
  fi
  record_status "stopped_by_signal"
  exit 143
}
trap on_signal INT TERM

declare -a idle_streak=(0 0)
checks=0
required_streak=6  # At least five elapsed minutes at one-minute spacing.
record_status "waiting pid=$$ required_idle_checks=$required_streak"
while true; do
  checks=$((checks + 1))
  for gpu in 0 1; do
    sample="$(nvidia-smi -i "$gpu" --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' || true)"
    apps="$(nvidia-smi -i "$gpu" --query-compute-apps=pid \
      --format=csv,noheader,nounits 2>/dev/null || true)"
    IFS=, read -r memory utilization <<< "$sample"
    if [[ "${memory:-}" =~ ^[0-9]+$ && "${utilization:-}" =~ ^[0-9]+$ &&
          -z "$apps" ]] && (( memory <= 1024 && utilization <= 10 )); then
      idle_streak[$gpu]=$((idle_streak[$gpu] + 1))
    else
      idle_streak[$gpu]=0
    fi
    if [[ "${E3_WAIT_DRY_RUN:-0}" == "1" ]]; then
      printf '%s gpu=%s memory=%s utilization=%s compute_apps=%s idle_streak=%s\n' \
        "$(date -u +%FT%TZ)" "$gpu" "${memory:-unknown}" "${utilization:-unknown}" \
        "$([[ -z "$apps" ]] && echo none || echo present)" "${idle_streak[$gpu]}"
    fi
  done
  record_status "waiting pid=$$ checks=$checks gpu0_idle=${idle_streak[0]} gpu1_idle=${idle_streak[1]}"
  if [[ "${E3_WAIT_DRY_RUN:-0}" == "1" ]]; then
    exit 0
  fi
  for gpu in 0 1; do
    if (( idle_streak[$gpu] < required_streak )); then
      continue
    fi
    record_status "launching gpu=$gpu pid=$$"
    printf '%s selected gpu=%s after %s idle checks\n' \
      "$(date -u +%FT%TZ)" "$gpu" "${idle_streak[$gpu]}"
    CONFIRM_SHARED_GPU=1 bash "$repo/reproduction/scripts/run_e3_pools_background.sh" \
      "$repo" "$base" "$output" "$models" "$gpu" &
    child_pid=$!
    set +e
    wait "$child_pid"
    result=$?
    set -e
    child_pid=""
    if (( result == 0 )); then
      record_status "pool_complete_e3a_running gpu=$gpu"
      python=/home/bkbs/miniforge3/envs/tabular-benchmark/bin/python
      direction_output=/mnt/nfs/bkbs/results/ZJU-Graduation-Project/e3-direction-confirm-v1
      audit_output=/mnt/nfs/bkbs/results/ZJU-Graduation-Project/e3-direction-audit-v1
      export PYTHONPATH="$repo/reproduction/src"
      export CUDA_VISIBLE_DEVICES=""
      export OMP_NUM_THREADS=1
      export OPENBLAS_NUM_THREADS=1
      export MKL_NUM_THREADS=1
      export NUMEXPR_NUM_THREADS=1
      nice -n 10 "$python" "$repo/reproduction/scripts/run_e3_direction.py" \
        --config "$repo/reproduction/configs/e3_direction_confirm_v1.yaml" \
        --registry "$output/pool_registry.csv" \
        --replica-manifest "$output/replica_manifest.json" \
        --output "$direction_output" >> "$control/e3a.log" 2>&1 || {
          record_status "failed stage=e3a_run"
          exit 5
        }
      record_status "e3a_complete_auditing gpu=$gpu"
      nice -n 10 "$python" "$repo/reproduction/scripts/analyze_e3_direction.py" \
        --input "$direction_output" --output "$audit_output" --expected-cases 6 \
        >> "$control/e3a.log" 2>&1 || {
          record_status "failed stage=e3a_audit"
          exit 6
        }
      record_status "complete gpu=$gpu e3a_audited=6"
      exit 0
    fi
    if (( result == 4 )); then
      # Race: another member occupied the card between our poll and final gate.
      record_status "waiting launch_race_gpu=$gpu"
      idle_streak=(0 0)
      break
    fi
    record_status "failed gpu=$gpu exit=$result"
    exit "$result"
  done
  sleep 60
done
