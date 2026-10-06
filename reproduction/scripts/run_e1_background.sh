#!/usr/bin/env bash
# One low-priority CPU worker on the shared server. No GPU allocation.
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 REPO DATA_ROOT OUTPUT_ROOT" >&2
  exit 2
fi
repo=$1
data_root=$2
output=$3
mkdir -p "$output"
exec 9>"$output/worker.lock"
if ! flock -n 9; then
  echo "Another E1 worker owns $output/worker.lock" >&2
  exit 3
fi

status() {
  printf '%s\t%s\n' "$(date -u +%FT%TZ)" "$1" > "$output/job-status.txt.partial"
  mv "$output/job-status.txt.partial" "$output/job-status.txt"
}
trap 'code=$?; if [[ $code -ne 0 ]]; then status "failed exit=$code"; fi' EXIT
status "running pid=$$"

export PYTHONPATH="$repo/reproduction/src"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
python=/home/bkbs/miniforge3/envs/tabular-benchmark/bin/python
config="$repo/reproduction/configs/e1_paired_screen_v1.yaml"
registry="$data_root/governance/pool_registry.csv"

for seed in 2026 2027 2028; do
  for table in adult credit abalone; do
    printf '%s\tstart\t%s\t%s\n' "$(date -u +%FT%TZ)" "$seed" "$table" >> "$output/progress.tsv"
    nice -n 10 "$python" "$repo/reproduction/scripts/run_e1_paired.py" \
      --config "$config" --registry "$registry" --output "$output" \
      --table "$table" --seed "$seed" >> "$output/worker.log" 2>&1
    printf '%s\tcomplete\t%s\t%s\n' "$(date -u +%FT%TZ)" "$seed" "$table" >> "$output/progress.tsv"
  done
done
status "complete"
