"""Validate the frozen M2-A contract bundle and run its CPU-only smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tabpollution.manifests.research_contract import validate_contract_bundle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contracts", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate_contract_bundle(args.contracts, args.config)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".partial")
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(args.output)
    print(json.dumps(result, ensure_ascii=False))

