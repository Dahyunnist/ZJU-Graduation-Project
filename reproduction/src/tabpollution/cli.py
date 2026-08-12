"""Command-line entry point for C0-C1 benchmark infrastructure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tabpollution.environment import capture_environment
from tabpollution.generators.smoke import run_adult_generator_smoke, validate_saved_adult_smokes
from tabpollution.generators.pilot import (
    load_pilot_config,
    run_adult_generator_pilot,
    validate_pilot_run,
)
from tabpollution.generators.preflight import generator_preflight
from tabpollution.mixing.commands import (
    build_smoke_bag_manifest,
    build_smoke_contamination_recipe,
    rebuild_smoke_bag,
)
from tabpollution.mixing.protocols import validate_protocol
from tabpollution.mixing.smoke import inspect_bag, run_c3_smoke
from tabpollution.pipeline import prepare_benchmark, validate_prepared_benchmark
from tabpollution.runs import aggregate_formal_runs
from tabpollution.algorithm_runthrough import run_all as run_algorithm_runthrough, recover_and_aggregate
from tabpollution.governance import (
    aggregate_governance_shards,
    build_shard_plan,
    run_governance_benchmark,
    run_governance_shard,
    run_governance_sharded,
    shard_queue,
    shard_status,
    validate_governance_setup,
)
from tabpollution.governance.pools import (
    build_governance_pools,
    preflight_pool_build,
    prepare_governance_sources,
)
from tabpollution.utils import write_json


def _default_config(benchmark: str) -> Path:
    return Path("configs") / f"{benchmark}.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tabpollution")
    commands = parser.add_subparsers(dest="command", required=True)

    data = commands.add_parser("data")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    prepare = data_commands.add_parser("prepare")
    prepare.add_argument("--config", type=Path, required=True)
    validate = data_commands.add_parser("validate")
    validate.add_argument("--benchmark", default="benchmark_v1")
    validate.add_argument("--config", type=Path)

    environment = commands.add_parser("environment")
    environment_commands = environment.add_subparsers(dest="environment_command", required=True)
    capture = environment_commands.add_parser("capture")
    capture.add_argument("--output", type=Path, default=Path("reports/environment_initial.txt"))

    generator = commands.add_parser("generator")
    generator_commands = generator.add_subparsers(dest="generator_command", required=True)
    generator_smoke = generator_commands.add_parser("smoke")
    generator_smoke.add_argument("--generator", required=True, choices=["GaussianCopula", "CTGAN", "TVAE"])
    generator_smoke.add_argument("--project-root", type=Path, default=Path("."))
    generator_validate = generator_commands.add_parser("validate")
    generator_validate.add_argument("--project-root", type=Path, default=Path("."))
    generator_validate.add_argument("--run-id")
    generator_preflight_parser = generator_commands.add_parser("preflight")
    generator_preflight_parser.add_argument("--config", type=Path, default=Path("configs/pilot_c2.yaml"))
    generator_preflight_parser.add_argument("--project-root", type=Path, default=Path("."))
    generator_preflight_parser.add_argument("--output", type=Path)
    generator_pilot = generator_commands.add_parser("pilot")
    generator_pilot.add_argument("--generator", required=True, choices=["GaussianCopula", "CTGAN", "TVAE"])
    generator_pilot.add_argument("--dataset", default="adult", choices=["adult"])
    generator_pilot.add_argument("--seed", type=int, default=2026, choices=[2026])
    generator_pilot.add_argument("--config", type=Path, default=Path("configs/pilot_c2.yaml"))
    generator_pilot.add_argument("--project-root", type=Path, default=Path("."))

    mixing = commands.add_parser("mixing")
    mixing_commands = mixing.add_subparsers(dest="mixing_command", required=True)
    mixing_smoke = mixing_commands.add_parser("smoke")
    mixing_smoke.add_argument("--generator", required=True, choices=["GaussianCopula", "CTGAN", "TVAE"])
    mixing_smoke.add_argument("--project-root", type=Path, default=Path("."))
    mixing_build = mixing_commands.add_parser("build")
    mixing_build.add_argument("--generator", required=True, choices=["GaussianCopula", "CTGAN", "TVAE"])
    mixing_build.add_argument("--condition", required=True, choices=["real_only", "real_append", "synthetic_append", "synthetic_replace"])
    mixing_build.add_argument("--proportion", required=True, type=float, choices=[0, .05, .10, .25, .50, .75, 1.0])
    mixing_build.add_argument("--output", type=Path, required=True)
    mixing_build.add_argument("--manifest-only", action="store_true")
    mixing_build.add_argument("--materialize", action="store_true")
    mixing_build.add_argument("--project-root", type=Path, default=Path("."))

    bags = commands.add_parser("bags")
    bags_commands = bags.add_subparsers(dest="bags_command", required=True)
    inspect = bags_commands.add_parser("inspect")
    inspect.add_argument("--generator", required=True)
    inspect.add_argument("--bag-id", required=True)
    inspect.add_argument("--project-root", type=Path, default=Path("."))
    bags_build = bags_commands.add_parser("build")
    bags_build.add_argument("--generator", required=True, choices=["GaussianCopula", "CTGAN", "TVAE"])
    bags_build.add_argument("--stage", required=True, choices=["calibration", "test"])
    bags_build.add_argument("--proportion", required=True, type=float, choices=[0, .05, .10, .25, .50, .75, 1.0])
    bags_build.add_argument("--bag-index", type=int, default=0)
    bags_build.add_argument("--output", type=Path, required=True)
    bags_build.add_argument("--manifest-only", action="store_true")
    bags_build.add_argument("--project-root", type=Path, default=Path("."))
    rebuild = bags_commands.add_parser("rebuild")
    rebuild.add_argument("--generator", required=True, choices=["GaussianCopula", "CTGAN", "TVAE"])
    rebuild.add_argument("--manifest", type=Path, required=True)
    rebuild.add_argument("--project-root", type=Path, default=Path("."))

    protocol = commands.add_parser("protocol")
    protocol_commands = protocol.add_subparsers(dest="protocol_command", required=True)
    protocol_validate = protocol_commands.add_parser("validate")
    protocol_validate.add_argument("--protocol", required=True, choices=["P1", "P2", "P3", "P4", "P5"])
    protocol_validate.add_argument("--manifest", type=Path, required=True)

    runs = commands.add_parser("runs")
    runs_commands = runs.add_subparsers(dest="runs_command", required=True)
    runs_aggregate = runs_commands.add_parser("aggregate")
    runs_aggregate.add_argument("--runs-dir", type=Path, default=Path("runs"))

    runthrough = commands.add_parser("runthrough")
    runthrough_commands = runthrough.add_subparsers(dest="runthrough_command", required=True)
    runthrough_all = runthrough_commands.add_parser("all")
    runthrough_all.add_argument("--project-root", type=Path, default=Path("."))
    runthrough_aggregate = runthrough_commands.add_parser("aggregate")
    runthrough_aggregate.add_argument("--project-root", type=Path, default=Path("."))

    governance = commands.add_parser(
        "governance",
        help="Run the unified detection-to-decision contamination benchmark",
    )
    governance_commands = governance.add_subparsers(dest="governance_command", required=True)
    governance_preflight = governance_commands.add_parser("preflight")
    governance_preflight.add_argument("--config", type=Path, required=True)
    governance_run = governance_commands.add_parser("run")
    governance_run.add_argument("--config", type=Path, required=True)
    shard_plan = governance_commands.add_parser("shard-plan")
    shard_plan.add_argument("--config", type=Path, required=True)
    shard_status_parser = governance_commands.add_parser("shard-status")
    shard_status_parser.add_argument("--config", type=Path, required=True)
    shard_queue_parser = governance_commands.add_parser("shard-queue")
    shard_queue_parser.add_argument("--config", type=Path, required=True)
    shard_queue_parser.add_argument("--seed", type=int, required=True)
    shard_queue_parser.add_argument("--resource-class", choices=["cpu", "gpu"], required=True)
    shard_run = governance_commands.add_parser("shard-run")
    shard_run.add_argument("--config", type=Path, required=True)
    shard_run.add_argument("--shard-id", required=True)
    shard_run.add_argument("--resume", action="store_true")
    shard_run.add_argument("--execution-device", choices=["cpu", "cuda"])
    sharded_run = governance_commands.add_parser("sharded-run")
    sharded_run.add_argument("--config", type=Path, required=True)
    sharded_run.add_argument("--resume", action="store_true")
    sharded_run.add_argument("--max-shards", type=int)
    shard_aggregate = governance_commands.add_parser("shard-aggregate")
    shard_aggregate.add_argument("--config", type=Path, required=True)
    source_prepare = governance_commands.add_parser("source-prepare")
    source_prepare.add_argument("--config", type=Path, required=True)
    pool_preflight = governance_commands.add_parser("pool-preflight")
    pool_preflight.add_argument("--config", type=Path, required=True)
    pool_build = governance_commands.add_parser("pool-build")
    pool_build.add_argument("--config", type=Path, required=True)
    pool_build.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "data" and args.data_command == "prepare":
        result = prepare_benchmark(args.config)
    elif args.command == "data" and args.data_command == "validate":
        result = validate_prepared_benchmark(args.config or _default_config(args.benchmark))
    elif args.command == "environment" and args.environment_command == "capture":
        capture_environment(args.output)
        result = {"environment_report": str(args.output)}
    elif args.command == "generator" and args.generator_command == "smoke":
        result = run_adult_generator_smoke(args.project_root, args.generator)
    elif args.command == "generator" and args.generator_command == "validate":
        result = (
            validate_pilot_run(args.project_root, args.run_id)
            if args.run_id
            else validate_saved_adult_smokes(args.project_root)
        )
    elif args.command == "generator" and args.generator_command == "preflight":
        load_pilot_config(args.project_root / args.config)
        result = generator_preflight(args.project_root)
        if args.output:
            write_json(result, args.project_root / args.output)
    elif args.command == "generator" and args.generator_command == "pilot":
        result = run_adult_generator_pilot(args.project_root, args.generator, args.config)
    elif args.command == "mixing" and args.mixing_command == "smoke":
        result = run_c3_smoke(args.project_root, args.generator)
    elif args.command == "mixing" and args.mixing_command == "build":
        if args.manifest_only and args.materialize:
            raise ValueError("Choose manifest-only or materialize, not both")
        result = build_smoke_contamination_recipe(
            args.project_root,
            args.generator,
            args.condition,
            args.proportion,
            args.output,
            materialize=args.materialize,
        )
    elif args.command == "bags" and args.bags_command == "inspect":
        result = inspect_bag(args.project_root, args.generator, args.bag_id)
    elif args.command == "bags" and args.bags_command == "build":
        result = build_smoke_bag_manifest(
            args.project_root,
            args.generator,
            args.stage,
            args.proportion,
            args.bag_index,
            args.output,
        )
    elif args.command == "bags" and args.bags_command == "rebuild":
        result = rebuild_smoke_bag(args.project_root, args.generator, args.manifest)
    elif args.command == "protocol" and args.protocol_command == "validate":
        result = validate_protocol(args.protocol, json.loads(args.manifest.read_text(encoding="utf-8")))
    elif args.command == "runs" and args.runs_command == "aggregate":
        result = aggregate_formal_runs(args.runs_dir)
    elif args.command == "runthrough" and args.runthrough_command == "all":
        result = run_algorithm_runthrough(args.project_root)
    elif args.command == "runthrough" and args.runthrough_command == "aggregate":
        result = recover_and_aggregate(args.project_root)
    elif args.command == "governance" and args.governance_command == "preflight":
        result = validate_governance_setup(args.config)
    elif args.command == "governance" and args.governance_command == "run":
        result = run_governance_benchmark(args.config)
    elif args.command == "governance" and args.governance_command == "shard-plan":
        result = build_shard_plan(args.config)
    elif args.command == "governance" and args.governance_command == "shard-status":
        result = shard_status(args.config)
    elif args.command == "governance" and args.governance_command == "shard-queue":
        result = shard_queue(
            args.config, seed=args.seed, resource_class=args.resource_class,
        )
    elif args.command == "governance" and args.governance_command == "shard-run":
        result = run_governance_shard(
            args.config, args.shard_id, resume=args.resume,
            execution_device=args.execution_device,
        )
    elif args.command == "governance" and args.governance_command == "sharded-run":
        result = run_governance_sharded(
            args.config, resume=args.resume, max_shards=args.max_shards,
        )
    elif args.command == "governance" and args.governance_command == "shard-aggregate":
        result = aggregate_governance_shards(args.config)
    elif args.command == "governance" and args.governance_command == "source-prepare":
        result = prepare_governance_sources(args.config)
    elif args.command == "governance" and args.governance_command == "pool-preflight":
        result = preflight_pool_build(args.config)
    elif args.command == "governance" and args.governance_command == "pool-build":
        result = build_governance_pools(args.config, resume=args.resume)
    else:
        raise AssertionError("Unreachable command")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
