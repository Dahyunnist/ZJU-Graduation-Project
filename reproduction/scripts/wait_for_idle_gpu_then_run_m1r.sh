#!/usr/bin/env bash
# Wait for sustained idle capacity, then run exactly one locked M1R workflow.
set -euo pipefail

repo=/mnt/nfs/bkbs/projects/ZJU-Graduation-Project
control=/mnt/nfs/bkbs/results/ZJU-Graduation-Project/m1r-launch-v1
mkdir -p "$control"
exec 9>"$control/wait.lock"
if ! flock -n 9; then
  echo "Another M1R GPU waiter is active" >&2
  exit 3
fi

record_status() {
  printf '%s\t%s\n' "$(date -u +%FT%TZ)" "$1" > "$control/wait-status.txt.partial"
  mv "$control/wait-status.txt.partial" "$control/wait-status.txt"
}

declare -a idle_streak=(0 0)
required_streak=6
checks=0
record_status "waiting pid=$$ required_idle_checks=$required_streak"
while true; do
  checks=$((checks + 1))
  for gpu in 0 1; do
    sample=$(nvidia-smi -i "$gpu" --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' || true)
    apps=$(nvidia-smi -i "$gpu" --query-compute-apps=pid \
      --format=csv,noheader,nounits 2>/dev/null || true)
    IFS=, read -r memory utilization <<< "$sample"
    if [[ "${memory:-}" =~ ^[0-9]+$ && "${utilization:-}" =~ ^[0-9]+$ && -z "$apps" ]] \
       && (( memory <= 1024 && utilization <= 10 )); then
      idle_streak[$gpu]=$((idle_streak[$gpu] + 1))
    else
      idle_streak[$gpu]=0
    fi
  done
  record_status "waiting pid=$$ checks=$checks gpu0_idle=${idle_streak[0]} gpu1_idle=${idle_streak[1]}"
  for gpu in 0 1; do
    if (( idle_streak[$gpu] < required_streak )); then continue; fi
    record_status "launching gpu=$gpu"
    set +e
    CONFIRM_SHARED_GPU=1 bash "$repo/reproduction/scripts/run_m1r_background.sh" "$repo" "$gpu"
    result=$?
    set -e
    if (( result == 0 )); then
      record_status "complete gpu=$gpu"
      exit 0
    fi
    if (( result == 4 )); then
      idle_streak=(0 0)
      record_status "waiting launch_race_gpu=$gpu"
      break
    fi
    record_status "failed gpu=$gpu exit=$result"
    exit "$result"
  done
  sleep 60
done
