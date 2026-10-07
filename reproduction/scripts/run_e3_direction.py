"""Preflight or run E3 external direction confirmation on a completed new pool."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tabpollution.studies.e3_direction import preflight, run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--replica-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output:
        result = run(args.config, args.registry, args.replica_manifest, args.output)
    else:
        result = preflight(args.config, args.registry, args.replica_manifest)
    print(json.dumps(result, ensure_ascii=False))
