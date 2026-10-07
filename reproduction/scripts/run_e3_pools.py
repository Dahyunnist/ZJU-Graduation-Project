"""Preflight or build generator repeats without changing the real split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tabpollution.studies.e3_pools import build, preflight


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--run", action="store_true", help="Fit GPU generators; default only checks inputs")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = (build(args.config, args.base_root, args.output_root, args.checkpoint_root,
                    resume=args.resume) if args.run else
              preflight(args.config, args.base_root, args.output_root, args.checkpoint_root))
    print(json.dumps(result, indent=2, ensure_ascii=False))
