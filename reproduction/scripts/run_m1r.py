"""Preflight, train pools, or run frozen E3B detection for M1R."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tabpollution.studies.m1r_repeats import (
    preflight,
    run_new_detections,
    train_new_instances,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--phase", choices=["preflight", "train", "detect"], required=True)
    args = parser.parse_args()
    common = (args.config, args.repository_root, args.base_root, args.dataset_root,
              args.checkpoint_root, args.results_root)
    if args.phase == "preflight":
        result = preflight(*common)
    elif args.phase == "train":
        if os.environ.get("M1R_GPU_TRAINING_APPROVED") != "1":
            parser.error("train requires M1R_GPU_TRAINING_APPROVED=1 after shared-GPU gate")
        result = train_new_instances(
            args.config, args.repository_root, args.base_root, args.dataset_root,
            args.checkpoint_root,
        )
    else:
        result = run_new_detections(
            args.config, args.base_root, args.dataset_root, args.checkpoint_root,
            args.results_root / "m1r-independent-repeats-v1",
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))

