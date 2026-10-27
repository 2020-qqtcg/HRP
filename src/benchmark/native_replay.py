"""Sample immutable, already-collected native trajectories without replaying prefixes."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import time
from pathlib import Path

from src.benchmark.native_experiment import SAMPLING_SCHEDULE, sample_task
from src.benchmark.native_gateway import NativeGateway, write_json
from src.benchmark.native_runtime import ROOT, source_fingerprints, verify_local_model_service, checkpoint_fingerprint
from src.hrp.native_measurement import MEASUREMENT_VERSION, fingerprint


def resume_config(config: dict, output: Path) -> dict:
    """Retain sample identities when only the replay driver has changed."""
    previous = json.loads((output / "manifest.json").read_text())
    ignored = {"fingerprint", "sampling_implementation"}
    if ({k: v for k, v in config.items() if k not in ignored}
            != {k: v for k, v in previous.items() if k not in ignored}):
        raise ValueError("Resume changes the frozen sampling configuration")
    if config["sampling_implementation"] != previous["sampling_implementation"]:
        snapshot = output / "sampling-implementation/src"
        if not snapshot.is_dir():
            raise ValueError("Resume needs the original sampling implementation snapshot")
        if source_fingerprints(measurement_root=snapshot)["measurement"] != previous["sampling_implementation"]:
            raise ValueError("Sampling implementation snapshot fingerprint mismatch")
        # The driver handles resource verification and dispatch. Actual sampler,
        # decoder, request construction and sample seed code must stay identical.
        for base, other in ((snapshot, ROOT / "src"), (ROOT / "src", snapshot)):
            for path in base.rglob("*"):
                if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
                    continue
                relative = path.relative_to(base)
                if relative == Path("benchmark/native_replay.py"):
                    continue
                counterpart = other / relative
                if not counterpart.is_file() or path.read_bytes() != counterpart.read_bytes():
                    raise ValueError(f"Resume changes sampling implementation: {relative}")
    return previous


def _eligible_paths(directory: Path) -> list[Path]:
    """Return recorded model-call boundaries eligible for native probing."""
    paths = []
    for path in sorted((directory / "wire").glob("*/record.json")):
        record = json.loads(path.read_text())
        if "messages" in record.get("request", {}):
            paths.append(path)
    return paths


def _select_paths(source: Path, relative: Path, manifest: dict,
                  limit: int | None) -> tuple[list[Path], dict]:
    """Choose a reproducible uniform subset while leaving the source untouched."""
    directory = source / relative
    all_paths = _eligible_paths(directory)
    if limit is None or len(all_paths) <= limit:
        selected = all_paths
        method = "all" if limit is None else "uniform_without_replacement_all"
        seed = None
    else:
        seed = int(hashlib.sha256(
            f"{manifest['fingerprint']}:{relative.as_posix()}".encode()
        ).hexdigest()[:16], 16)
        selected = [all_paths[index] for index in sorted(
            random.Random(seed).sample(range(len(all_paths)), limit)
        )]
        method = "uniform_without_replacement"
    total = len(all_paths)
    return selected, {
        "trajectory": relative.as_posix(),
        "total_recorded_states": total,
        "selected_recorded_states": len(selected),
        "sampling_method": method,
        "selection_seed": seed,
        "inclusion_probability": len(selected) / total if total else None,
        "horvitz_thompson_scale": total / len(selected) if selected else None,
        "selected_calls": [path.parent.name for path in selected],
    }


async def replay(args):
    source = args.experiment.resolve()
    manifest = json.loads((source / "manifest.json").read_text())
    if "native" not in manifest["interfaces"]:
        raise ValueError("Native HRP cannot consume historical shared-Bash trajectories")
    selected = []
    for result_path in sorted(source.glob("*/*/native/repeat-*/*/result.json")):
        result = json.loads(result_path.read_text())
        if args.harnesses and result["harness"] not in args.harnesses:
            continue
        if args.benchmarks and result["benchmark"] not in args.benchmarks:
            continue
        if args.task_ids and result["task_id"] not in args.task_ids:
            continue
        if args.repeats and result["repeat"] not in args.repeats:
            continue
        selected.append((result_path, result))
    if not selected:
        raise ValueError("No native trajectories match the requested selection")
    config = json.loads(json.dumps(manifest))
    if args.ceiling:
        if manifest.get("rollout_model_role") != "teacher":
            raise ValueError("Ceiling probes require teacher-generated trajectories")
        config["purpose"] = "teacher_occupancy_regularized_ceiling_proxy"
        config["hrp"]["gamma"] = 1
    service_config = json.loads(json.dumps(config))
    if args.gpu_ids is not None:
        service_config["runtime"].update(gpu_ids=args.gpu_ids, data_parallel_size=len(args.gpu_ids))
    execution_verification = verify_local_model_service(service_config)
    # GPU placement and replica count are execution provenance. Preserve the
    # original fingerprint used for all cached sample identities and seeds.
    config["model_service_verification"] = json.loads(json.dumps(execution_verification))
    config["model_service_verification"]["gpu_ids"] = config["runtime"]["gpu_ids"]
    config["model_service_verification"]["checked_launch_flags"]["--data-parallel-size"] = config["runtime"]["data_parallel_size"]
    if manifest.get("target_checkpoint_files"):
        current_checkpoint = await asyncio.to_thread(checkpoint_fingerprint, Path(config["target"]["checkpoint"]))
        if current_checkpoint != manifest["target_checkpoint_files"]:
            raise ValueError("Target checkpoint content changed since trajectory collection")
    current_sources = source_fingerprints()
    archived_source = source / "implementation/src"
    decoder_source = (source_fingerprints(measurement_root=archived_source)["measurement"]
                      if archived_source.is_dir() else current_sources["measurement"])
    if decoder_source != manifest["sources"]["measurement"]:
        raise ValueError("Recorded adapter source differs from its manifest; restore the pinned implementation before sampling")
    config["native_decoder_source_verification"] = {"status": "verified", "fingerprint": decoder_source,
        "path": str(archived_source) if archived_source.is_dir() else str(ROOT / "src")}
    for harness in manifest["harnesses"]:
        if current_sources[harness] != manifest["sources"][harness]:
            raise ValueError(f"{harness} vendor source/runtime changed; restore its pinned version before sampling")
    dependencies = manifest["sources"].get("protocol_dependencies", {})
    for dependency in ("uv.lock", "pyproject.toml"):
        if dependency in dependencies and dependencies[dependency] != current_sources["protocol_dependencies"][dependency]:
            raise ValueError(f"{dependency} changed; restore the recorded native SDK environment before sampling")
    config["dependency_pin_validation"] = "verified" if dependencies else "legacy acceptance trace without dependency hashes"
    config["source_trajectory_fingerprint"] = config.pop("fingerprint")
    config["mode"] = "frozen_native_replay"
    config["sampling_implementation"] = current_sources["measurement"]
    config["hrp"].update(n_target=args.n_target, n_teacher=args.n_teacher)
    config["hrp"]["measurement_version"] = MEASUREMENT_VERSION
    config["hrp"]["sampling_schedule"] = SAMPLING_SCHEDULE
    config["concurrency"].setdefault("native_sample_workers", 40)
    config["trajectory_selection"] = [str(path.parent.relative_to(source)) for path, _ in selected]
    config["validation_only"] = args.validation_only
    config["validation_states_per_task"] = args.states_per_task if args.validation_only else None
    config["fingerprint"] = fingerprint(config)
    if args.resume_probe:
        output = args.resume_probe.resolve()
        if output.parent != args.output.resolve():
            raise ValueError("--resume-probe must belong to --output")
        config = resume_config(config, output)
    else:
        output = args.output.resolve() / config["fingerprint"][:20]
    output.mkdir(parents=True, exist_ok=True)
    if not args.resume_probe:
        write_json(output / "manifest.json", config)
    with (output / "execution-history.jsonl").open("a") as stream:
        stream.write(json.dumps({"time": time.time(), "resumed": bool(args.resume_probe),
            "actual_model_service_verification": execution_verification,
            "actual_sampling_implementation": current_sources["measurement"],
            "sampling_config_fingerprint": config["fingerprint"]}) + "\n")
    write_json(args.output.resolve() / "latest.json", {"experiment": str(output), "fingerprint": config["fingerprint"]})
    selections = {}
    selected_paths = {}
    for result_path, _ in selected:
        relative = result_path.parent.relative_to(source)
        paths, selection = _select_paths(source, relative, manifest,
                                         args.max_states_per_trajectory)
        selections[relative.as_posix()] = selection
        selected_paths[relative.as_posix()] = paths if args.max_states_per_trajectory is not None else None
    write_json(output / "state-selection.json", {
        "schema": "native-replay-state-selection-v1",
        "source_experiment": str(source),
        "sampling_unit": "native model-call boundary",
        "sampling_method": "uniform_without_replacement",
        "max_states_per_trajectory": args.max_states_per_trajectory,
        "trajectories": selections,
        "source_states_retained": True,
        "source_retention": "wire records and agent/states remain in the trajectory experiment",
    })
    credentials = json.loads((ROOT / "api.json").read_text()) if (ROOT / "api.json").exists() else {}
    key = os.environ.get("OPENROUTER_API_KEY") or credentials.get("openrouter_api_key") or credentials.get("key") or credentials.get("api_key")
    if not key:
        raise ValueError("OPENROUTER_API_KEY is required")
    gateway = NativeGateway({**config["target"], "api_key": "EMPTY"}, {**config["teacher"], "api_key": key},
                            local_concurrency=config["concurrency"]["local_requests"],
                            teacher_concurrency=config["concurrency"]["teacher_requests"],
                            native_worker_concurrency=config["concurrency"]["native_sample_workers"])
    await gateway.start()
    try:
        summaries = []
        groups = {}
        for result_path, result in selected:
            directory = result_path.parent
            relative = directory.relative_to(source)
            destination = output / relative
            destination.mkdir(parents=True, exist_ok=True)
            chosen = selected_paths[relative.as_posix()]
            summary = await sample_task(directory, gateway, config, output=destination,
                validation_state_limit=args.states_per_task if args.validation_only else None,
                state_paths=chosen)
            summaries.append({"benchmark": result["benchmark"], "harness": result["harness"],
                              "task_id": result["task_id"], "repeat": result["repeat"], "summary": str(destination / "summary.json"),
                              "states": len(summary["states"])})
            key = f"{result['benchmark']}/{result['harness']}/{result['interface']}"
            groups.setdefault(key, []).append(summary["diagnostics"])
            write_json(output / "summary.json", {"measurement_version": MEASUREMENT_VERSION,
                "trajectories": summaries, "validation_only": args.validation_only, "groups": groups,
                "aggregation": "per-trajectory diagnostics retained; no implicit weighting by trajectory length",
                "peak_requests": gateway.peak, "harness_ranking": None, "success_prediction": None})
    finally:
        await gateway.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume-probe", type=Path,
                        help="Resume this existing probe directory, preserving sample identities")
    parser.add_argument("--gpu-ids", nargs="+", type=int,
                        help="Actual data-parallel GPUs, recorded separately from frozen sample identity")
    parser.add_argument("--harnesses", nargs="+", choices=("pi", "deepagents", "deepseek", "terminus-2", "mini-swe-agent"))
    parser.add_argument("--benchmarks", nargs="+", choices=("tb21", "tb_lite"))
    parser.add_argument("--task-ids", nargs="+")
    parser.add_argument("--repeats", nargs="+", type=int, help="Original repeat IDs to sample, e.g. --repeats 1")
    parser.add_argument("--n-target", type=int, default=128)
    parser.add_argument("--n-teacher", type=int, default=64)
    parser.add_argument("--ceiling", action="store_true",
                        help="Probe teacher trajectories for finite-horizon ceiling scoring")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--states-per-task", type=int, default=1)
    parser.add_argument("--max-states-per-trajectory", type=int,
                        help="Uniformly sample at most this many recorded boundaries per trajectory")
    args = parser.parse_args()
    if args.gpu_ids is not None and (min(args.gpu_ids) < 0 or len(set(args.gpu_ids)) != len(args.gpu_ids)):
        parser.error("--gpu-ids must contain distinct nonnegative IDs")
    if min(args.n_target, args.n_teacher, args.states_per_task) < 1:
        parser.error("Sample counts and validation states must be positive")
    if args.max_states_per_trajectory is not None and args.max_states_per_trajectory < 1:
        parser.error("--max-states-per-trajectory must be positive")
    asyncio.run(replay(args))


if __name__ == "__main__":
    main()
