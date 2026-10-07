#!/usr/bin/env bash
# Sequential low-priority CPU execution on the shared server.
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 REPOSITORY DATA_ROOT OUTPUT_ROOT" >&2
  exit 2
fi
research_repo=$1
research_data=$2
research_output=$3
mkdir -p "$research_output"
exec 9>"$research_output/worker.lock"
if ! flock -n 9; then
  echo "E1B worker already active" >&2
  exit 3
fi
record_status() {
  printf '%s\t%s\n' "$(date -u +%FT%TZ)" "$1" > "$research_output/job-status.txt.partial"
  mv "$research_output/job-status.txt.partial" "$research_output/job-status.txt"
}
trap 'exit_code=$?; if [[ $exit_code -ne 0 ]]; then record_status "failed exit=$exit_code"; fi' EXIT
record_status "running pid=$$"

export PYTHONPATH="$research_repo/reproduction/src"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
research_python=/home/bkbs/miniforge3/envs/tabular-benchmark/bin/python
research_config="$research_repo/reproduction/configs/e1b_probability_diagnostic_v1.yaml"
research_registry="$research_data/governance/pool_registry.csv"

for research_seed in 2026 2027 2028; do
  for research_case in abalone:TVAE credit:TVAE adult:CTGAN; do
    research_table=${research_case%%:*}
    research_generator=${research_case##*:}
    printf '%s\tstart\t%s\t%s\n' "$(date -u +%FT%TZ)" "$research_seed" "$research_case" >> "$research_output/progress.tsv"
    nice -n 10 "$research_python" "$research_repo/reproduction/scripts/run_e1b.py" \
      --config "$research_config" --registry "$research_registry" --output "$research_output" \
      --table "$research_table" --generator "$research_generator" --seed "$research_seed" \
      >> "$research_output/worker.log" 2>&1
    printf '%s\tcomplete\t%s\t%s\n' "$(date -u +%FT%TZ)" "$research_seed" "$research_case" >> "$research_output/progress.tsv"
  done
done
record_status "complete"
