"""Probability tables for Figure 4: one frozen H0 actor under native H0--H5."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from hrp_miniwob.adapters.miniwob_harnesses import Adapter, LanguagePolicy
from hrp_miniwob.adapters.native_miniwob import write_json
from hrp_miniwob.benchmarks import miniwob_harnesses as harness_study
from hrp_miniwob.benchmarks import miniwob_section51 as section51
from hrp_miniwob.benchmarks import miniwob_section51_measurement_v2 as measurement_v2
from hrp_miniwob.benchmarks.miniwob_meta_harness import digest, model_files
from hrp_miniwob.harnesses.miniwob_harnesses import Candidate
from hrp_miniwob.suite import reachable_states

ROOT = section51.ROOT
EXPERIMENT = ROOT / "environment/artifacts/v2/figure4_h0_h5_prediction_2026-09-17"
DOCS = ROOT / "docs/figure4_h0_h5_prediction_2026-09-17"
BRIEF = Path(__file__).resolve()
CALIBRATION_SOURCE = measurement_v2.SOURCE
TEST_SOURCE = measurement_v2.EXPERIMENT
SOURCE = TEST_SOURCE
BASE_MODEL = section51.BASE_MODEL
H0_ADAPTER = section51.H0_ADAPTER
HARNESSES = tuple(f"H{index}" for index in range(6))
GPU_HARNESSES = ("H1", "H2", "H4", "H5")
PHYSICAL_GPUS = (0, 1, 2, 3)
GAMMA, TAU, HORIZON = .95, .1, 10


def read(path):
    return json.loads(Path(path).read_text())


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def candidate_path(harness: str) -> Path:
    return harness_study.candidate_path(harness)


def source_paths():
    paths = [
        Path(__file__),
        ROOT / "environment/hrp_miniwob/benchmarks/miniwob_figure4_harnesses_analysis.py",
        ROOT / "environment/hrp_miniwob/orchestration/run_miniwob_figure4_harnesses.py",
        ROOT / "environment/hrp_miniwob/validation/check_miniwob_figure4_harnesses.py",

        ROOT / "environment/hrp_miniwob/adapters/miniwob_harnesses.py",
        ROOT / "environment/hrp_miniwob/harnesses/miniwob_harnesses.py",
        ROOT / "environment/hrp_miniwob/benchmarks/miniwob_section51.py",
        ROOT / "environment/hrp_miniwob/benchmarks/miniwob_section51_measurement_v2.py",
        ROOT / "environment/hrp_miniwob/benchmarks/miniwob_section51_analysis.py",
        ROOT / "environment/hrp_miniwob/oracle.py",
        ROOT / "environment/training/uv.lock",
        *[candidate_path(harness) for harness in HARNESSES],
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Figure 4 source set is incomplete: " + ", ".join(missing))
    return paths


def source_hashes():
    return {str(path.relative_to(ROOT)): digest(path) for path in source_paths()}


def proposal():
    return {
        "schema_version": 1,
        "task_id": "figure4_h0_h5_prediction_2026-09-17",
        "study_source": str(BRIEF),
        "study_source_sha256": digest(BRIEF),
        "source_experiments": {
            "calibration": str(CALIBRATION_SOURCE), "test": str(TEST_SOURCE),
        },
        "source_manifest_sha256": {
            "calibration": digest(CALIBRATION_SOURCE / "suite/manifest.json"),
            "test": digest(TEST_SOURCE / "suite/manifest.json"),
        },
        "source_config_sha256": {
            "calibration": digest(CALIBRATION_SOURCE / "config.json"),
            "test": digest(TEST_SOURCE / "config.json"),
        },
        "source_model": {
            "training_harness": "H0", "checkpoint_update": 1024,
            "training_seed": 42, "adapter_path": str(H0_ADAPTER),
            "adapter_sha256": model_files(H0_ADAPTER),
        },
        "harnesses": list(HARNESSES),
        "numbering": read(section51.E1 / "harness_numbering.json"),
        "splits": {"calibration": 72, "test": 576, "tasks_per_family": {"calibration": 24, "test": 192}},
        "estimator": {"m": 256, "N": 1024, "alpha": .5, "pools_per_test_task": 1},
        "rollouts_per_task_harness": 32,
        "gamma": GAMMA, "tau": TAU, "horizon": HORIZON,
        "execution": "native deterministic H0--H5; zero explicit action cost",
        "reference": "frozen Qwen3.5-2B reads each native harness prompt independently",
        "physical_gpus": list(PHYSICAL_GPUS),
        "training": "none; frozen E1 H0 update 1024 actor; source seed 42",
    }


def prepare(experiment=EXPERIMENT):
    experiment = Path(experiment).resolve()
    experiment.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    path = experiment / "proposal.json"
    value = proposal()
    if path.exists() and read(path) != value:
        raise ValueError("Existing Figure 4 proposal differs from the fixed protocol")
    if not path.exists():
        write_json(path, value)
    manifests = {
        "calibration": read(CALIBRATION_SOURCE / "suite/manifest.json"),
        "test": read(TEST_SOURCE / "suite/manifest.json"),
    }
    if any(manifest.get("status") != "complete" for manifest in manifests.values()):
        raise RuntimeError("Measurement V2 calibration/test source suite is incomplete")
    split_counts = {split: sum(entry["split"] == split for entry in manifest["entries"])
                    for split, manifest in manifests.items()}
    if split_counts != {"calibration": 72, "test": 576}:
        raise AssertionError(f"Unexpected reused split counts: {split_counts}")
    write_json(experiment / "preparation.json", {
        "status": "complete", "source_manifest_sha256": value["source_manifest_sha256"],
        "source_config_sha256": value["source_config_sha256"], "split_counts": split_counts,
        "task_reuse": "exact V2 tasks and split; no regeneration",
        "prepared_at_utc": datetime.now(UTC).isoformat(),
    })
    write_json(experiment / "progress.json", {
        "status": "prepared", "completed_cells": 0, "total_cells": 12,
        "updated_at_utc": datetime.now(UTC).isoformat(),
    })
    return value


def models(split):
    source = CALIBRATION_SOURCE if split == "calibration" else TEST_SOURCE
    return section51.load_models(source, split)


def split_source(split):
    return CALIBRATION_SOURCE if split == "calibration" else TEST_SOURCE


def cell_directory(experiment, split, harness):
    return Path(experiment) / "probabilities" / split / harness


def task_table_path(output, model):
    return Path(output) / "tasks" / f"{model.task.key}.npz"


def cell_complete(experiment, split, harness):
    output = cell_directory(experiment, split, harness)
    completion = output / "completion.json"
    return completion.exists() and read(completion).get("accounting_sha256") == digest(output / "accounting.json")


def _save_table(path, model, harness, graph, bank, log_policy, log_reference, prompts, codes, metadata_extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not np.allclose(np.exp(log_policy[bank]).sum(1), 1, atol=1e-12, rtol=0):
        raise AssertionError("Actor table is not normalized on reachable states")
    if not np.allclose(np.exp(log_reference[bank]).sum(1), 1, atol=1e-12, rtol=0):
        raise AssertionError("Reference table is not normalized on reachable states")
    prompt_hashes = np.asarray([
        sha256_bytes(json.dumps(prompt, ensure_ascii=False, sort_keys=True).encode())
        for prompt in prompts
    ])
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream, bank=np.asarray(bank), log_policy=log_policy, log_reference=log_reference,
            prompt_sha256=prompt_hashes, semantic_actions=np.asarray(graph.task.actions),
            semantic_codes=np.asarray(codes[0]), next_state=graph.next_state,
            task_rewards=graph.task_rewards, initial_state=np.asarray(graph.initial_state),
        )
    temporary.replace(path)
    metadata = {
        "task_seed": model.task.seed, "family": model.task.family,
        "content_sha256": model.task.fingerprint, "harness": harness,
        "states": len(bank), "graph_states": len(graph.next_state),
        "actions": len(graph.task.actions), "table_sha256": digest(path),
        **metadata_extra,
    }
    write_json(path.with_suffix(".json"), metadata)
    return metadata


def _import_task(experiment, split, harness, model, output):
    source = split_source(split)
    base = section51.load_table(source, split, f"D_base_{harness}", model)
    actor = section51.load_table(source, split, f"D_h3_update_0000_{harness}", model)
    for key in ("bank", "prompt_sha256", "semantic_actions", "semantic_codes", "next_state", "task_rewards", "initial_state"):
        if not np.array_equal(base[key], actor[key]):
            raise AssertionError(f"V2 native H0/H3 base and actor disagree for {key}")
    spec = section51.cell_specs(split)[f"D_h3_update_0000_{harness}"]
    graph, bank, rendered = section51.graph_and_prompts(model, spec)
    if not np.array_equal(base["bank"], bank):
        raise AssertionError("Imported reachable-state bank changed")
    if not np.array_equal(base["next_state"], graph.next_state) or not np.array_equal(base["task_rewards"], graph.task_rewards):
        raise AssertionError("Imported native graph changed")
    prompts = [row.messages for row in rendered]
    codes = [row.semantic_codes for row in rendered]
    expected_hashes = np.asarray([
        sha256_bytes(json.dumps(prompt, ensure_ascii=False, sort_keys=True).encode())
        for prompt in prompts
    ])
    if not np.array_equal(base["prompt_sha256"], expected_hashes):
        raise AssertionError("Imported native prompt changed")
    return _save_table(
        task_table_path(output, model), model, harness, graph, bank,
        section51.full_log_policy(actor), section51.full_log_policy(base), prompts, codes,
        {"reused": True, "source_actor_table_sha256": digest(section51.task_table_path(section51.cell_directory(source, split, f"D_h3_update_0000_{harness}"), model)),
         "source_reference_table_sha256": digest(section51.task_table_path(section51.cell_directory(source, split, f"D_base_{harness}"), model)),
         "forward_state_rows": 0, "prompt_tokens": 0, "forward_microbatches": 0},
    )


def _measure_task(bound, model, harness, output):
    path = task_table_path(output, model)
    metadata_path = path.with_suffix(".json")
    if path.exists() and metadata_path.exists():
        metadata = read(metadata_path)
        if metadata["content_sha256"] != model.task.fingerprint or metadata["table_sha256"] != digest(path):
            raise ValueError("Resumed Figure 4 task table failed identity validation")
        return metadata
    before = (bound.policy.forward_observations, bound.policy.forward_prompt_tokens, bound.policy.forward_microbatches)
    started = time.monotonic()
    graph, log_policy = bound.measure(model)
    bank = reachable_states(graph)
    prompts, codes = zip(*(bound.messages(graph.state(int(index))) for index in bank), strict=True)
    after = (bound.policy.forward_observations, bound.policy.forward_prompt_tokens, bound.policy.forward_microbatches)
    return _save_table(
        path, model, harness, graph, bank, log_policy, np.log(graph.reference()), prompts, codes,
        {"reused": False, "wall_seconds": time.monotonic() - started,
         "forward_state_rows": after[0] - before[0], "prompt_tokens": after[1] - before[1],
         "forward_microbatches": after[2] - before[2],
         "reference_requests": bound.frozen.requests, "reference_forwards": bound.frozen.forwards,
         "reviewer_requests": bound.reviewer.requests, "reviewer_forwards": bound.reviewer.forwards},
    )


def measure_cell(experiment, split, harness, device="cuda:0"):
    experiment = Path(experiment).resolve()
    if harness not in HARNESSES or split not in ("calibration", "test"):
        raise ValueError("Unknown Figure 4 split or harness")
    source = split_source(split)
    if digest(source / "suite/manifest.json") != read(experiment / "proposal.json")["source_manifest_sha256"][split]:
        raise ValueError("Reused V2 task manifest changed")
    output = cell_directory(experiment, split, harness)
    output.mkdir(parents=True, exist_ok=True)
    candidate = Candidate(candidate_path(harness))
    identity = {
        "proposal_sha256": digest(experiment / "proposal.json"), "split": split,
        "harness": harness, "candidate_sha256": candidate.sha256,
        "actor_adapter": str(H0_ADAPTER), "actor_model_files_sha256": model_files(H0_ADAPTER),
        "base_model_files_sha256": model_files(BASE_MODEL), "source_seed": 42,
        "source_v2_config_sha256": digest(source / "config.json"),
        "device": "reused_cpu" if harness in {"H0", "H3"} else device,
    }
    identity_path = output / "identity.json"
    if identity_path.exists() and read(identity_path) != identity:
        raise ValueError("Figure 4 probability directory belongs to another immutable cell")
    if not identity_path.exists():
        write_json(identity_path, identity)
    if cell_complete(experiment, split, harness):
        print(json.dumps(read(output / "completion.json")), flush=True)
        return
    task_models = models(split)
    rows, started = [], time.monotonic()
    policy = bound = None
    try:
        if harness not in {"H0", "H3"}:
            policy = LanguagePolicy(
                str(BASE_MODEL), seed=42, device=device, adapter_path=str(H0_ADAPTER),
                trainable=False, rank=16,
            )
            if any(parameter.requires_grad for parameter in policy.model.parameters()):
                raise AssertionError("Figure 4 actor must be frozen")
            policy.model.eval()
            bound = Adapter(policy, candidate, policy, policy, experiment / "frozen" / harness)
        for number, model in enumerate(task_models, 1):
            row = (_import_task(experiment, split, harness, model, output)
                   if harness in {"H0", "H3"} else _measure_task(bound, model, harness, output))
            rows.append(row)
            write_json(output / "progress.json", {
                "status": "running", "completed_tasks": number, "total_tasks": len(task_models),
                "updated_at_utc": datetime.now(UTC).isoformat(),
            })
        accounting = {
            "status": "complete", "split": split, "harness": harness, "tasks": len(rows),
            "states": sum(row["states"] for row in rows),
            "forward_state_rows": sum(row.get("forward_state_rows", 0) for row in rows),
            "prompt_tokens": sum(row.get("prompt_tokens", 0) for row in rows),
            "forward_microbatches": sum(row.get("forward_microbatches", 0) for row in rows),
            "wall_seconds": time.monotonic() - started,
            "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "reused_from_v2": harness in {"H0", "H3"},
            "model_metadata": None if policy is None else policy.metadata,
        }
        write_json(output / "accounting.json", accounting)
        completion = {
            "status": "complete", "identity_sha256": digest(identity_path),
            "accounting_sha256": digest(output / "accounting.json"),
            "task_table_hashes_sha256": sha256_bytes(json.dumps(
                {row["task_seed"]: row["table_sha256"] for row in rows}, sort_keys=True
            ).encode()),
            "completed_at_utc": datetime.now(UTC).isoformat(),
        }
        write_json(output / "completion.json", completion)
        write_json(output / "progress.json", {"status": "complete", "completed_tasks": len(rows), "total_tasks": len(rows)})
        print(json.dumps(completion), flush=True)
    finally:
        if bound is not None:
            bound.close()
        del bound, policy
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_table(experiment, split, harness, model):
    path = task_table_path(cell_directory(experiment, split, harness), model)
    metadata = read(path.with_suffix(".json"))
    if digest(path) != metadata["table_sha256"] or metadata["content_sha256"] != model.task.fingerprint:
        raise ValueError(f"Figure 4 table failed checksum/identity validation: {path}")
    with np.load(path) as saved:
        return {key: saved[key].copy() for key in saved.files}


def full_policy(table, field):
    shape = table["next_state"].shape
    if table[field].shape == shape:
        return np.exp(table[field])
    if table[field].shape != (len(table["bank"]), shape[1]):
        raise ValueError(f"Unexpected saved probability shape for {field}: {table[field].shape}")
    result = np.full(shape, -np.log(shape[1]), dtype=np.float64)
    result[table["bank"]] = table[field]
    return np.exp(result)


def audit_tables(experiment, split):
    task_models = models(split)
    checks = states = 0
    for harness in HARNESSES:
        if not cell_complete(experiment, split, harness):
            raise RuntimeError(f"Incomplete Figure 4 probability cell: {split}/{harness}")
        for model in task_models:
            table = load_table(experiment, split, harness, model)
            bank = table["bank"]
            if len(np.unique(bank)) != len(bank) or np.any(bank < 0) or np.any(bank >= len(table["next_state"])):
                raise AssertionError("Invalid reachable-state bank")
            for field in ("log_policy", "log_reference"):
                if not np.all(np.isfinite(table[field])) or not np.allclose(np.exp(table[field]).sum(1), 1, atol=1e-12, rtol=0):
                    raise AssertionError(f"Non-normalized {field}")
            if len(table["semantic_actions"]) != table["next_state"].shape[1]:
                raise AssertionError("Semantic action table shape mismatch")
            if int(table["initial_state"]) not in set(map(int, bank)):
                raise AssertionError("Initial state is absent from the measured bank")
            checks += 1
            states += len(bank)
    result = {"status": "passed", "split": split, "cells": 6, "tasks_per_cell": len(task_models),
              "task_cell_checks": checks, "reachable_rows_checked": states}
    output = Path(experiment) / "acceptance"
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / f"{split}_probability_tables.json", result)
    return result


def seal_inputs(experiment=EXPERIMENT):
    experiment = Path(experiment).resolve()
    calibration = read(experiment / "calibration/summary.json")
    if calibration.get("status") != "complete":
        raise RuntimeError("Figure 4 calibration is incomplete")
    audit = audit_tables(experiment, "calibration")
    config = proposal() | {
        "sealed_at_utc": datetime.now(UTC).isoformat(),
        "proposal_sha256": digest(experiment / "proposal.json"),
        "calibration_sha256": digest(experiment / "calibration/summary.json"),
        "calibration_audit": audit,
        "source_sha256": source_hashes(),
        "cell_identity_sha256": {
            harness: digest(cell_directory(experiment, "calibration", harness) / "identity.json")
            for harness in HARNESSES
        },
    }
    path = experiment / "config.json"
    if path.exists():
        old = read(path)
        config["sealed_at_utc"] = old["sealed_at_utc"]
        if old != config:
            raise ValueError("Sealed Figure 4 inputs changed")
        return old
    snapshot = experiment / "source_snapshot"
    for source in source_paths():
        target = snapshot / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    write_json(path, config)
    write_json(experiment / "manifest.json", {
        "status": "sealed", "config_sha256": digest(path),
        "test_order": "all 3456 finite predictions sealed before exact scoring",
        "training_seed_statement": "frozen E1 H0 source checkpoint uses seed 42 only",
    })
    return config


def verify_seal(experiment=EXPERIMENT):
    experiment = Path(experiment).resolve()
    config = read(experiment / "config.json")
    if digest(experiment / "proposal.json") != config["proposal_sha256"]:
        raise ValueError("Figure 4 proposal changed after sealing")
    if digest(experiment / "calibration/summary.json") != config["calibration_sha256"]:
        raise ValueError("Figure 4 calibration changed after sealing")
    if source_hashes() != config["source_sha256"]:
        raise ValueError("Figure 4 source changed after sealing")
    if model_files(H0_ADAPTER) != config["source_model"]["adapter_sha256"]:
        raise ValueError("Frozen H0 checkpoint changed")
    for split, source in (("calibration", CALIBRATION_SOURCE), ("test", TEST_SOURCE)):
        if digest(source / "suite/manifest.json") != config["source_manifest_sha256"][split]:
            raise ValueError(f"Reused {split} task manifest changed")
    return config


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    measure = sub.add_parser("measure-cell")
    measure.add_argument("--split", choices=("calibration", "test"), required=True)
    measure.add_argument("--harness", choices=HARNESSES, required=True)
    measure.add_argument("--device", default="cuda:0")
    audit = sub.add_parser("audit")
    audit.add_argument("--split", choices=("calibration", "test"), required=True)
    sub.add_parser("seal-inputs")
    parser.add_argument("--experiment", type=Path, default=EXPERIMENT)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    torch.set_num_threads(2)
    if args.command == "prepare":
        print(json.dumps(prepare(args.experiment), indent=2))
    elif args.command == "measure-cell":
        measure_cell(args.experiment, args.split, args.harness, args.device)
    elif args.command == "audit":
        print(json.dumps(audit_tables(args.experiment, args.split), indent=2))
    else:
        print(json.dumps(seal_inputs(args.experiment), indent=2))


if __name__ == "__main__":
    main()
