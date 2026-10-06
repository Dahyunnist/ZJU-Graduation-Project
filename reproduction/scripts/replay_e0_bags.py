"""Reconstruct historical sampling, never fit models; not saved-mask evidence."""
import argparse
import json
from pathlib import Path

import pandas as pd

from reanalyze_e0 import sha, dump_json, BASE
from tabpollution.governance.data import RegistrySource, exact_mixture, sample_rows
from tabpollution.governance.pipeline import _split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads((args.run / "resolved_config.json").read_text())
    output = args.analysis / "sampling_reconstruction.csv"
    if output.exists():
        raise FileExistsError(output)
    registry = Path(cfg["registry_path"])
    source = RegistrySource(registry)
    base = pd.read_csv(args.analysis / "base_bags.csv")
    selected = base.loc[base.utility_scheduled.eq(True)]
    records = []
    cached = {}
    for row in selected.to_dict("records"):
        protocol = cfg["protocols"][row["protocol"]]
        ti = protocol["test_tables"].index(row["test_table"])
        gi = protocol["test_generators"].index(row["test_generator"])
        seed = int(row["seed"])
        key = (seed, row["protocol"], row["test_table"], row["test_generator"])
        table = source.table(row["test_table"])
        if key not in cached:
            rs = _split(table.real, seed + ti * 113)
            ss = _split(table.synthetic[row["test_generator"]], seed + ti*113 + gi*997 + 31)
            cached[key] = (rs["downstream_train"], ss["downstream_train"], rs["final_test"])
        real, synth, test = cached[key]
        bag_seed = seed + ti*100003 + gi*1009 + int(row["nominal_prevalence"]*1000)*17 + int(row["bag_index"])
        bag = exact_mixture(real, synth, cfg["bag_size"], row["nominal_prevalence"], bag_seed, mode=row["contamination_mode"])
        clean = sample_rows(real, cfg["bag_size"], bag_seed+71)
        if len(bag) != row["bag_size"] or abs(bag.source_label.mean()-row["true_prevalence"]) > 1e-12:
            raise ValueError("Reconstructed bag disagrees with saved counts")
        actual_real = bag.loc[bag.source_label.eq(0)]
        real_ids, test_ids = set(actual_real.record_id), set(test.record_id)
        records.append({k: row[k] for k in BASE} | {
            "reconstruction_not_saved_indices": True,
            "n_real_pool": len(real), "n_synthetic_pool": len(synth),
            "n_bag": len(bag), "n_unique_record_ids": bag.record_id.nunique(),
            "duplicate_fraction": 1-bag.record_id.nunique()/len(bag),
            "bag_task_classes": bag[table.target_column].nunique(),
            "oracle_remaining_n": len(actual_real),
            "oracle_task_classes": actual_real[table.target_column].nunique(),
            "real_test_id_overlap": len(real_ids & test_ids),
            "clean_baseline_real_overlap_n": len(real_ids & set(clean.record_id)),
            "clean_baseline_unique_n": clean.record_id.nunique(),
        })
    pd.DataFrame(records).to_csv(output, index=False)
    registry_df = pd.read_csv(registry)
    paths = {registry}
    for col in ("real_path", "synthetic_path"):
        paths.update((registry.parent / p).resolve() for p in registry_df[col])
    dump_json(args.analysis / "sampling_reconstruction_provenance.json", {
        "script_sha256": sha(__file__), "rows": len(records),
        "inputs": [{"path": str(p), "sha256": sha(p)} for p in sorted(paths)],
        "limitation": "Replayed using current unchanged historical sampling implementation and registered pools; not original saved row indices or deletion masks.",
    })
    print(f"Reconstructed {len(records)} scheduled utility bags without training", flush=True)


if __name__ == "__main__":
    main()
