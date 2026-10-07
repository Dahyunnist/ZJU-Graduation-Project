#!/usr/bin/env bash
# One low-priority CPU-only worker; completed bags are skipped on restart.
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
  echo "E2 worker already active" >&2
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
research_config="$research_repo/reproduction/configs/e2_minimal_decision_chain_v1.yaml"
research_registry="$research_data/governance/pool_registry.csv"
for research_seed in 2026 2027 2028; do
  printf '%s\tstart\t%s\n' "$(date -u +%FT%TZ)" "$research_seed" >> "$research_output/progress.tsv"
  nice -n 10 "$research_python" "$research_repo/reproduction/scripts/run_e2.py" \
    --config "$research_config" --registry "$research_registry" \
    --output "$research_output" --seed "$research_seed" >> "$research_output/worker.log" 2>&1
  printf '%s\tcomplete\t%s\n' "$(date -u +%FT%TZ)" "$research_seed" >> "$research_output/progress.tsv"
done
record_status "complete"
