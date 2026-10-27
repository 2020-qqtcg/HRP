"""Frozen-policy readout equivalence experiment on native MiniWoB++ H0.

This module owns data preparation, immutable configuration, GPU probability
measurement, deterministic controls, and exact CPU analysis.  Every GPU cell
contains one model, one rendering, and one split and is independently resumable.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import time
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from hrp_miniwob.adapters.miniwob_readout import (
    RENDERINGS,
    ReadoutRenderer,
    align_encoded_columns,
    encoded_column_order,
    mapping_record,
)
from hrp_miniwob.adapters.miniwob_v2 import LanguagePolicy
from hrp_miniwob.adapters.native_miniwob import write_json
from hrp_miniwob.benchmarks.miniwob_meta_harness import digest, model_files
from hrp_miniwob.oracle import evaluate, solve
from hrp_miniwob.suite import reachable_states
from hrp_miniwob.v2 import FAMILIES
from hrp_miniwob.v2.browser import Browser, Task
from hrp_miniwob.v2.model import compile_task
from hrp_miniwob.v2.suite import is_ood, verify_success_path


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "environment/artifacts/v2/readout_equivalence_2026-09-15"
DOCS = ROOT / "environment/docs/v2/readout_equivalence_2026-09-15"
BASE_MODEL = Path("models/Qwen3.5-2B")
H0_ADAPTER = ROOT / "environment/artifacts/v2/qwen35_2b_h0_h7_exactpg_seed42_2026-09-11/runs/exactpg_H0/checkpoint_001024/actor/adapter"
H3_ADAPTER = ROOT / "environment/artifacts/v2/figure3_h0_h3_exactpg_2026-09-14/runs/seed42_H3/checkpoint_001024/actor/adapter"
MODEL_SPECS = {
    "base": None,
    "h0_final": H0_ADAPTER,
    "h3_adapt_final": H3_ADAPTER,
}
GAMMA, TAU, HORIZON = .95, .1, 10
BOOTSTRAP_SEED, BOOTSTRAP_SAMPLES = 4205, 2000
START_SEED = 41_100_000
SPLIT_QUOTAS = {"calibration": 24, "test": 192}


def read(path):
    return json.loads(Path(path).read_text())


def source_paths():
    paths = [
        Path(__file__),
        ROOT / "environment/hrp_miniwob/adapters/miniwob_readout.py",
        ROOT / "environment/hrp_miniwob/orchestration/run_miniwob_readout.py",
        ROOT / "environment/hrp_miniwob/validation/check_miniwob_readout.py",
        ROOT / "environment/hrp_miniwob/benchmarks/miniwob_readout_report.py",


        ROOT / "environment/hrp_miniwob/adapters/miniwob_v2.py",
        ROOT / "environment/hrp_miniwob/harnesses/miniwob_harnesses.py",
        ROOT / "environment/hrp_miniwob/harnesses/miniwob_h0.py",
        ROOT / "environment/hrp_miniwob/v2/model.py",
        ROOT / "environment/hrp_miniwob/v2/browser.py",
        ROOT / "environment/hrp_miniwob/v2/suite.py",
        ROOT / "environment/hrp_miniwob/oracle.py",
        ROOT / "environment/hrp_miniwob/measurements.py",
        ROOT / "environment/training/uv.lock",
    ]
    return paths


def source_hashes():
    return {str(path.relative_to(ROOT)): digest(path) for path in source_paths()}


def model_hashes():
    return {
        "base": model_files(BASE_MODEL),
        "h0_final": model_files(H0_ADAPTER),
        "h3_adapt_final": model_files(H3_ADAPTER),
    }


def proposal():
    return {
        "schema_version": 1,
        "created_for": "Chapter 5 readout and correspondence study",
        "families": list(FAMILIES),
        "splits_per_family": SPLIT_QUOTAS,
        "start_seed": START_SEED,
        "family_seed_offset": 1_000_000,
        "renderings": list(RENDERINGS),
        "models": {key: str(value) if value else str(BASE_MODEL) for key, value in MODEL_SPECS.items()},
        "horizon": HORIZON,
        "gamma": GAMMA,
        "tau": TAU,
        "explicit_cost": 0.0,
        "reference_protocol": "shared semantic q0 measured from base model under canonical R0",
        "natural_reference_analysis": "base model semantic policy measured separately under each rendering",
        "decoder": "complete categorical distribution over reversible one-token menu; temperature 1",
        "dtype": "bfloat16 model forward; float64 probabilities and exact control calculations",
        "microbatch_states": 8,
        "bootstrap": {"samples": BOOTSTRAP_SAMPLES, "seed": BOOTSTRAP_SEED, "unit": "paired task within family"},
        "physical_gpus": [0, 1, 2, 3],
        "training": "none; all three policies frozen",
    }


def _collect_exclusions(experiment):
    contents, seeds, manifests = set(), set(), {}
    for path in sorted((ROOT / "environment/artifacts").rglob("manifest.json")):
        if experiment in path.parents or any(part in {"source_snapshot", "runtime_source", "source", "vendor"} for part in path.parts):
            continue
        try:
            record = read(path)
        except (json.JSONDecodeError, OSError):
            continue
        used = False
        if record.get("version") == "v2" and isinstance(record.get("entries"), list):
            for entry in record["entries"]:
                if "content_sha256" in entry:
                    contents.add(entry["content_sha256"])
                    seeds.add(int(entry["seed"]))
                    used = True
        elif isinstance(record.get("tasks"), list):
            for entry in record["tasks"]:
                if not isinstance(entry, dict) or "json" not in entry:
                    continue
                task_path = path.parent / entry["json"]
                if not task_path.exists():
                    continue
                payload = read(task_path).get("task", {})
                if "labels" in payload and "instruction" in payload:
                    native = Task(payload["seed"], "click-checkboxes", payload["instruction"], {"labels": payload["labels"]}, 0)
                    contents.add(native.fingerprint)
                    seeds.add(int(payload["seed"]))
                    used = True
        if used:
            manifests[str(path)] = digest(path)
    return {
        "manifest_sha256": manifests,
        "content_sha256": sorted(contents),
        "seeds": sorted(seeds),
    }


def prepare_data(experiment=EXPERIMENT):
    """Create fresh content-disjoint calibration/test tasks with live replay."""
    experiment = Path(experiment).resolve()
    experiment.mkdir(parents=True, exist_ok=True)
    proposal_path = experiment / "proposal.json"
    if proposal_path.exists():
        if read(proposal_path) != proposal():
            raise ValueError("Existing readout proposal differs from the fixed design")
    else:
        write_json(proposal_path, proposal())
    if (experiment / "suite/manifest.json").exists():
        return read(experiment / "suite/manifest.json")

    exclusion_path = experiment / "exclusions.json"
    if exclusion_path.exists():
        exclusions = read(exclusion_path)
    else:
        exclusions = _collect_exclusions(experiment)
        write_json(exclusion_path, exclusions)
    for path, expected in exclusions["manifest_sha256"].items():
        if digest(path) != expected:
            raise ValueError(f"An exclusion source changed during preparation: {path}")
    forbidden_contents = set(exclusions["content_sha256"])
    forbidden_seeds = set(exclusions["seeds"])

    suite = experiment / "suite"
    (suite / "tasks").mkdir(parents=True, exist_ok=True)
    journal = experiment / "task_journal.jsonl"
    entries = [json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []
    seen = {entry["content_sha256"] for entry in entries}
    if seen & forbidden_contents or len(seen) != len(entries):
        raise AssertionError("Resumed task journal violates content exclusions")

    with Browser(horizon=HORIZON) as browser:
        for family_index, family in enumerate(FAMILIES):
            counts = Counter(entry["split"] for entry in entries if entry["family"] == family)
            old = [entry["seed"] for entry in entries if entry["family"] == family]
            seed = max(old) + 1 if old else START_SEED + family_index * 1_000_000
            scanned = 0
            while any(counts[split] < quota for split, quota in SPLIT_QUOTAS.items()):
                if scanned >= 100_000:
                    raise RuntimeError(f"Fresh IID task scan exhausted for {family}")
                task = browser.reset(seed, family).task
                seed += 1
                scanned += 1
                if task.seed in forbidden_seeds or task.fingerprint in forbidden_contents | seen or is_ood(task):
                    continue
                model = compile_task(task, HORIZON)
                success_steps = verify_success_path(browser, model)
                split = next(name for name, quota in SPLIT_QUOTAS.items() if counts[name] < quota)
                relative = f"tasks/{task.key}.json"
                path = suite / relative
                write_json(path, asdict(task))
                entry = {
                    "family": family,
                    "seed": task.seed,
                    "split": split,
                    "json": relative,
                    "sha256": digest(path),
                    "content_sha256": task.fingerprint,
                    "verified_success_steps": success_steps,
                    "actions": len(task.actions),
                    "physical_states": task.physical_states,
                    "reachable_states": len(reachable_states(model)),
                }
                with journal.open("a") as stream:
                    stream.write(json.dumps(entry) + "\n")
                    stream.flush()
                entries.append(entry)
                seen.add(task.fingerprint)
                counts[split] += 1
                write_json(experiment / "preparation.json", {
                    "status": "running", "family": family, "counts": dict(counts), "tasks": len(entries),
                })
            print(json.dumps({"family": family, "counts": dict(counts)}), flush=True)

    expected = len(FAMILIES) * sum(SPLIT_QUOTAS.values())
    if len(entries) != expected or len(seen) != expected:
        raise AssertionError("Prepared suite has the wrong number of unique tasks")
    manifest = {
        "version": "v2",
        "status": "complete",
        "profile": "readout_equivalence",
        "horizon": HORIZON,
        "families": list(FAMILIES),
        "entries": entries,
        "exclusions_sha256": digest(exclusion_path),
        "validation": "Every task has a live native success replay; all retained artifact manifests were excluded before scanning",
    }
    write_json(suite / "manifest.json", manifest)
    splits = {split: [entry["seed"] for entry in entries if entry["split"] == split] for split in SPLIT_QUOTAS}
    write_json(experiment / "splits.json", splits)
    write_json(experiment / "preparation.json", {
        "status": "complete",
        "counts": {split: len(seeds) for split, seeds in splits.items()},
        "content_overlap": 0,
        "seed_overlap": 0,
        "excluded_manifests": len(exclusions["manifest_sha256"]),
    })
    return manifest


def load_models(experiment, split):
    manifest = read(Path(experiment) / "suite/manifest.json")
    if manifest["status"] != "complete" or manifest["version"] != "v2":
        raise ValueError("Readout suite is incomplete")
    models = []
    for entry in manifest["entries"]:
        if entry["split"] != split:
            continue
        path = Path(experiment) / "suite" / entry["json"]
        if digest(path) != entry["sha256"]:
            raise ValueError(f"Prepared task changed: {path}")
        task = Task(**read(path))
        if task.fingerprint != entry["content_sha256"]:
            raise ValueError("Task content fingerprint changed")
        models.append(compile_task(task, HORIZON))
    if len(models) != len(FAMILIES) * SPLIT_QUOTAS[split]:
        raise ValueError(f"Wrong {split} task count")
    return models


def validate_model_specs():
    hashes = model_hashes()
    for key, adapter in MODEL_SPECS.items():
        if adapter is None:
            continue
        config = read(adapter / "adapter_config.json")
        if Path(config["base_model_name_or_path"]).resolve() != BASE_MODEL.resolve():
            raise ValueError(f"{key} was not trained from the fixed base model")
        if config["r"] != 16 or config["lora_alpha"] != 32:
            raise ValueError(f"{key} is not the sealed rank-16 adapter")
    return hashes


def seal(experiment=EXPERIMENT):
    experiment = Path(experiment).resolve()
    if read(experiment / "preparation.json")["status"] != "complete":
        raise RuntimeError("Data preparation must finish before sealing")
    acceptance = read(experiment / "acceptance/gpu_preflight.json")
    if acceptance["status"] != "passed":
        raise RuntimeError("GPU calibration preflight must pass before sealing")
    config = proposal() | {
        "sealed_at_utc": datetime.now(UTC).isoformat(),
        "suite_manifest_sha256": digest(experiment / "suite/manifest.json"),
        "splits_sha256": digest(experiment / "splits.json"),
        "exclusions_sha256": digest(experiment / "exclusions.json"),
        "model_files_sha256": validate_model_specs(),
        "source_sha256": source_hashes(),
        "acceptance_sha256": {
            "cpu": digest(experiment / "acceptance/cpu.json"),
            "gpu_preflight": digest(experiment / "acceptance/gpu_preflight.json"),
        },
    }
    path = experiment / "config.json"
    if path.exists():
        old = read(path)
        config["sealed_at_utc"] = old["sealed_at_utc"]
        if old != config:
            raise ValueError("Sealed readout configuration changed")
        return old
    snapshot = experiment / "source_snapshot"
    for source in source_paths():
        target = snapshot / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    write_json(path, config)
    write_json(experiment / "manifest.json", {
        "status": "sealed",
        "config_sha256": digest(path),
        "primary_split": "test",
        "test_blinding": "renderings, models, statistics and source frozen before any test cell",
        "training_seed_statement": "H0 source training and H3 adaptation used seed 42; model conditions are not independent training seeds",
    })
    return config


def read_config(experiment=EXPERIMENT):
    experiment = Path(experiment).resolve()
    config = read(experiment / "config.json")
    checks = {
        "suite/manifest.json": config["suite_manifest_sha256"],
        "splits.json": config["splits_sha256"],
        "exclusions.json": config["exclusions_sha256"],
    }
    for relative, expected in checks.items():
        if digest(experiment / relative) != expected:
            raise ValueError(f"Sealed input changed: {relative}")
    if source_hashes() != config["source_sha256"]:
        raise ValueError("Readout source differs from the sealed snapshot")
    if model_hashes() != config["model_files_sha256"]:
        raise ValueError("A frozen model or adapter changed after sealing")
    return config


def load_policy(model_id, device):
    if model_id not in MODEL_SPECS:
        raise ValueError(f"Unknown frozen model: {model_id}")
    adapter = MODEL_SPECS[model_id]
    policy = LanguagePolicy(
        str(BASE_MODEL), seed=42, device=device,
        adapter_path=str(adapter) if adapter else None,
        trainable=False, rank=16,
    )
    if any(parameter.requires_grad for parameter in policy.model.parameters()):
        raise AssertionError("Readout evaluation loaded trainable parameters")
    policy.model.eval()
    return policy


def task_policy_path(output, model):
    return Path(output) / "tasks" / f"{model.task.key}.npz"


def measure_task(policy, model, rendering_id, output):
    path = task_policy_path(output, model)
    meta_path = path.with_suffix(".json")
    if path.exists() and meta_path.exists():
        meta = read(meta_path)
        if meta["content_sha256"] != model.task.fingerprint:
            raise ValueError("Resumed task output has different public content")
        return meta
    path.parent.mkdir(parents=True, exist_ok=True)
    renderer = ReadoutRenderer(rendering_id)
    bank = reachable_states(model)
    rendered = [renderer.render(model.state(int(index))) for index in bank]
    prompts = [item.messages for item in rendered]
    codes = [item.semantic_codes for item in rendered]
    before = (policy.forward_observations, policy.forward_prompt_tokens, policy.forward_microbatches)
    started = time.monotonic()
    with torch.no_grad():
        lp_bank = policy.batch_log_probabilities(prompts, codes).detach().cpu().numpy()
    if not np.all(np.isfinite(lp_bank)) or not np.allclose(np.exp(lp_bank).sum(1), 1, atol=1e-12, rtol=0):
        raise AssertionError("Actor did not return a complete normalized menu distribution")
    temporary = path.with_name(path.name + f".{__import__('os').getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            bank=bank,
            log_policy=lp_bank,
            prompt_sha256=np.asarray([item.prompt_sha256 for item in rendered]),
            semantic_actions=np.asarray(model.task.actions),
            semantic_codes=np.asarray(codes[0]),
        )
    temporary.replace(path)
    after = (policy.forward_observations, policy.forward_prompt_tokens, policy.forward_microbatches)
    meta = {
        "task_seed": model.task.seed,
        "family": model.task.family,
        "content_sha256": model.task.fingerprint,
        "rendering_id": rendering_id,
        "states": len(bank),
        "actions": len(model.task.actions),
        "mapping": rendered[0].mapping,
        "wall_seconds": time.monotonic() - started,
        "forward_state_rows": after[0] - before[0],
        "prompt_tokens": after[1] - before[1],
        "forward_microbatches": after[2] - before[2],
        "policy_npz_sha256": digest(path),
    }
    write_json(meta_path, meta)
    return meta


def evaluate_cell(experiment, output, model_id, rendering_id, split, device):
    experiment, output = Path(experiment).resolve(), Path(output).resolve()
    config = read_config(experiment)
    output.mkdir(parents=True, exist_ok=True)
    identity = {
        "config_sha256": digest(experiment / "config.json"),
        "model_id": model_id,
        "model_files_sha256": config["model_files_sha256"][model_id],
        "rendering_id": rendering_id,
        "mapping_by_menu_size": {str(n): mapping_record(n, rendering_id) for n in range(1, 8)},
        "split": split,
        "device": device,
    }
    identity_path = output / "identity.json"
    if identity_path.exists() and read(identity_path) != identity:
        raise ValueError("Evaluation directory belongs to another fixed cell")
    if not identity_path.exists():
        write_json(identity_path, identity)
    if (output / "completion.json").exists():
        print(json.dumps(read(output / "completion.json")), flush=True)
        return
    policy = load_policy(model_id, device)
    started = time.monotonic()
    rows = []
    try:
        models = load_models(experiment, split)
        for number, model in enumerate(models, 1):
            rows.append(measure_task(policy, model, rendering_id, output))
            write_json(output / "progress.json", {
                "status": "running", "completed_tasks": number, "total_tasks": len(models),
                "updated_at_utc": datetime.now(UTC).isoformat(),
            })
        accounting = {
            "status": "complete",
            "model_id": model_id,
            "rendering_id": rendering_id,
            "split": split,
            "tasks": len(rows),
            "states": sum(row["states"] for row in rows),
            "forward_state_rows": sum(row["forward_state_rows"] for row in rows),
            "prompt_tokens": sum(row["prompt_tokens"] for row in rows),
            "forward_microbatches": sum(row["forward_microbatches"] for row in rows),
            "wall_seconds": time.monotonic() - started,
            "dtype": policy.metadata["dtype"],
            "decoder_temperature": policy.metadata["decoder_temperature"],
            "policy_metadata": policy.metadata,
        }
        write_json(output / "accounting.json", accounting)
        completion = {
            "status": "complete",
            "identity_sha256": digest(identity_path),
            "accounting_sha256": digest(output / "accounting.json"),
            "task_metadata_sha256": {str(task_policy_path(output, model).with_suffix('.json').relative_to(output)): digest(task_policy_path(output, model).with_suffix('.json')) for model in models},
            "completed_at_utc": datetime.now(UTC).isoformat(),
        }
        write_json(output / "completion.json", completion)
        write_json(output / "progress.json", {"status": "complete", "completed_tasks": len(models), "total_tasks": len(models)})
        print(json.dumps(accounting), flush=True)
    finally:
        del policy
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def select_repeat_states(model):
    bank = reachable_states(model)
    if len(bank) <= 16:
        return bank
    return bank[np.linspace(0, len(bank) - 1, 16, dtype=int)]


def repeat_forward(experiment, output, model_id, device):
    experiment, output = Path(experiment).resolve(), Path(output).resolve()
    read_config(experiment)
    output.mkdir(parents=True, exist_ok=True)
    completion = output / "completion.json"
    if completion.exists():
        print(completion.read_text(), flush=True)
        return
    policy = load_policy(model_id, device)
    renderer = ReadoutRenderer("R0")
    maxima = {"log_probability": 0.0, "semantic_tv": 0.0, "centered_readout": 0.0}
    tasks, states, started = [], 0, time.monotonic()
    try:
        for number, model in enumerate(load_models(experiment, "calibration"), 1):
            selected = select_repeat_states(model)
            rendered = [renderer.render(model.state(int(index))) for index in selected]
            prompts = [item.messages for item in rendered]
            codes = [item.semantic_codes for item in rendered]
            trials = []
            with torch.no_grad():
                for _ in range(5):
                    trials.append(policy.batch_log_probabilities(prompts, codes).detach().cpu().numpy())
            stacked = np.stack(trials)
            baseline = stacked[0]
            centered = TAU * (stacked - stacked.mean(axis=2, keepdims=True))
            maxima["log_probability"] = max(maxima["log_probability"], float(np.max(np.abs(stacked - baseline))))
            maxima["semantic_tv"] = max(maxima["semantic_tv"], float(np.max(.5 * np.abs(np.exp(stacked) - np.exp(baseline)).sum(2))))
            maxima["centered_readout"] = max(maxima["centered_readout"], float(np.max(np.abs(centered - centered[0]))))
            task_path = output / "tasks" / f"{model.task.key}.npz"
            task_path.parent.mkdir(parents=True, exist_ok=True)
            with task_path.open("wb") as stream:
                np.savez_compressed(stream, states=selected, log_policy=stacked)
            tasks.append({"family": model.task.family, "content_sha256": model.task.fingerprint, "states": len(selected), "sha256": digest(task_path)})
            states += len(selected)
            write_json(output / "progress.json", {"status": "running", "completed_tasks": number, "total_tasks": 72})
        report = {
            "status": "complete", "model_id": model_id, "tasks": len(tasks), "states_per_repeat": states,
            "repeats": 5, "maxima": maxima, "wall_seconds": time.monotonic() - started,
            "interpretation": "deterministic numerical repeatability; not training-seed or action-sampling variance",
            "tasks_detail": tasks,
        }
        write_json(output / "metrics.json", report)
        write_json(completion, {"status": "complete", "metrics_sha256": digest(output / "metrics.json")})
        write_json(output / "progress.json", {"status": "complete", "completed_tasks": 72, "total_tasks": 72})
        print(json.dumps(report), flush=True)
    finally:
        del policy
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def policy_table(cell, model):
    path = task_policy_path(cell, model)
    meta = read(path.with_suffix(".json"))
    if digest(path) != meta["policy_npz_sha256"]:
        raise ValueError(f"Saved probability table changed: {path}")
    with np.load(path) as table:
        bank = table["bank"].copy()
        lp = table["log_policy"].copy()
        actions = tuple(str(x) for x in table["semantic_actions"])
        codes = tuple(str(x) for x in table["semantic_codes"])
        prompt_hashes = tuple(str(x) for x in table["prompt_sha256"])
    if actions != tuple(model.task.actions):
        raise ValueError("Saved action columns are not in semantic H0 order")
    return bank, lp, codes, prompt_hashes


def centered(values):
    values = np.asarray(values, dtype=np.float64)
    return values - values.mean(axis=1, keepdims=True)


def independent_solution(model, reference_semantic, rendering_id):
    """Solve in visible canonical-code order, then decode to semantic order."""
    order = encoded_column_order(len(model.task.actions), rendering_id)
    solution = solve(
        model.next_state[:, order],
        model.task_rewards[:, order],
        np.asarray(reference_semantic)[:, order],
        GAMMA,
        TAU,
    )
    return {
        "policy": align_encoded_columns(solution.policy, rendering_id),
        "log_policy": align_encoded_columns(solution.log_policy, rendering_id),
        "energy": align_encoded_columns(solution.energy, rendering_id),
        "value": solution.value,
    }


def exact_metrics(model, actor_lp, reference, optimal):
    actor = np.exp(actor_lp)
    result = evaluate(model.next_state, model.task_rewards, reference, actor, model.initial_state, GAMMA, TAU)
    visits = result.occupancy / (1 - GAMMA)
    log_reference = np.log(reference)
    discounted_reward = float(visits @ np.sum(actor * model.task_rewards, axis=1))
    discounted_kl = float(TAU * visits @ np.sum(actor * (actor_lp - log_reference), axis=1))
    mismatch = float(TAU * visits @ np.sum(actor * (actor_lp - optimal["log_policy"]), axis=1))
    success = np.sum(actor * model.task_rewards, axis=1)
    for state, successors in enumerate(model.next_state):
        live = successors >= 0
        success[state] += actor[state, live] @ success[successors[live]]
    ceiling = float(optimal["value"][model.initial_state])
    return {
        "ceiling": ceiling,
        "regularized_return": result.return_value,
        "mismatch": mismatch,
        "success_probability": float(success[model.initial_state]),
        "discounted_task_reward": discounted_reward,
        "discounted_kl_penalty": discounted_kl,
        "return_decomposition_error": abs(result.return_value - discounted_reward + discounted_kl),
        "gap_identity_error": abs(ceiling - result.return_value - mismatch),
        "nonterminal_occupancy_mass": float(result.occupancy.sum()),
    }


def _full_table(model, bank, bank_values):
    values = np.full(model.next_state.shape, -np.log(len(model.task.actions)), dtype=np.float64)
    values[bank] = bank_values
    return values


def analyze_task(model, actor_tables, base_tables):
    bank0, q0_bank, _, hashes0 = base_tables["R0"]
    if not np.array_equal(bank0, reachable_states(model)):
        raise AssertionError("Base R0 table does not cover the fixed reachable bank")
    q0_lp = _full_table(model, bank0, q0_bank)
    q0 = np.exp(q0_lp)
    common_solutions = {rendering: independent_solution(model, q0, rendering) for rendering in RENDERINGS}
    target0 = TAU * centered(common_solutions["R0"]["log_policy"])
    actor0_lp = _full_table(model, actor_tables["R0"][0], actor_tables["R0"][1])
    actor0_energy = TAU * centered(actor0_lp)
    e0 = actor0_energy - target0
    rows, natural_rows, state_arrays = {}, {}, {
        "bank": bank0,
        "target_R0": target0[bank0],
        "e0": e0[bank0],
    }

    base_ceiling = common_solutions["R0"]["value"][model.initial_state]
    for rendering in RENDERINGS:
        bank, lp_bank, codes, hashes = actor_tables[rendering]
        if not np.array_equal(bank, bank0):
            raise AssertionError("Renderings use different fixed state banks")
        actor_lp = _full_table(model, bank, lp_bank)
        actor = np.exp(actor_lp)
        optimum = common_solutions[rendering]
        target = TAU * centered(optimum["log_policy"])
        actor_energy = TAU * centered(actor_lp)
        g = target0 - target
        eta = actor_energy - actor0_energy
        deployment_error = actor_energy - target
        tv = .5 * np.abs(actor - np.exp(actor0_lp)).sum(1)
        argmax = np.argmax(actor, axis=1) != np.argmax(np.exp(actor0_lp), axis=1)
        raw_delta = np.max(np.abs(actor_lp - actor0_lp), axis=1)
        pairwise = np.max(eta, axis=1) - np.min(eta, axis=1)
        l2 = np.linalg.norm(eta, axis=1)
        pair_bound = 2 * TAU * raw_delta
        l2_bound = TAU * np.sqrt(actor.shape[1]) * raw_delta
        metrics = exact_metrics(model, actor_lp, q0, optimum)
        actor_evaluation = evaluate(model.next_state, model.task_rewards, q0, actor, model.initial_state, GAMMA, TAU)
        metrics.update({
            "target_mse": float(np.mean(g[bank] ** 2)),
            "readout_mse": float(np.mean(eta[bank] ** 2)),
            "semantic_tv": float(np.mean(tv[bank])),
            "max_semantic_tv": float(np.max(tv[bank])),
            "argmax_change_rate": float(np.mean(argmax[bank])),
            "delta_tok_mean": float(np.mean(raw_delta[bank])),
            "delta_tok_max": float(np.max(raw_delta[bank])),
            "pairwise_drift_max": float(np.max(pairwise[bank])),
            "pairwise_bound_max_violation": float(np.max(pairwise[bank] - pair_bound[bank])),
            "pairwise_bound_mean_slack": float(np.mean(pair_bound[bank] - pairwise[bank])),
            "l2_drift_max": float(np.max(l2[bank])),
            "l2_bound_max_violation": float(np.max(l2[bank] - l2_bound[bank])),
            "l2_bound_mean_slack": float(np.mean(l2_bound[bank] - l2[bank])),
            "deployment_identity_max_error": float(np.max(np.abs(deployment_error[bank] - (e0[bank] + eta[bank] + g[bank])))),
            "ceiling_equivalence_error": abs(metrics["ceiling"] - base_ceiling),
            "optimal_policy_equivalence_max_error": float(np.max(np.abs(optimum["policy"][bank] - common_solutions["R0"]["policy"][bank]))),
            "prompt_hash_equal_to_r0_rate": float(np.mean(np.asarray(hashes) == np.asarray(hashes0))),
            "menu_codes": list(codes),
        })
        rows[rendering] = metrics
        state_arrays[f"g_{rendering}"] = g[bank]
        state_arrays[f"eta_{rendering}"] = eta[bank]
        state_arrays[f"deployment_error_{rendering}"] = deployment_error[bank]
        state_arrays[f"discounted_occupancy_{rendering}"] = actor_evaluation.occupancy

        natural_bank, natural_lp_bank, _, _ = base_tables[rendering]
        natural_reference = np.exp(_full_table(model, natural_bank, natural_lp_bank))
        natural_optimal = independent_solution(model, natural_reference, rendering)
        natural = exact_metrics(model, actor_lp, natural_reference, natural_optimal)
        natural_target = TAU * centered(natural_optimal["log_policy"])
        natural.update({
            "target_mse_vs_R0": float(np.mean((target0[bank] - natural_target[bank]) ** 2)),
            "readout_mse_vs_R0": metrics["readout_mse"],
        })
        natural_rows[rendering] = natural

    # Deterministic rename after R0 selection: permute output labels, decode the
    # inverse mapping, and recover the exact same semantic table.
    rename_perm = np.asarray(mapping_record(len(model.task.actions), "R4")["semantic_to_code_index"])
    renamed = np.empty_like(actor0_lp)
    renamed[:, rename_perm] = actor0_lp
    recovered = renamed[:, rename_perm]
    rename_error = float(np.max(np.abs(recovered - actor0_lp)))
    wrong = renamed.copy()
    wrong_error = float(np.max(np.abs(wrong[bank] - actor0_lp[bank])))
    rename_metrics = exact_metrics(model, recovered, q0, common_solutions["R0"])
    baseline_metrics = exact_metrics(model, actor0_lp, q0, common_solutions["R0"])
    rename_metric_errors = {
        key: abs(rename_metrics[key] - baseline_metrics[key])
        for key in ("regularized_return", "mismatch", "success_probability", "ceiling")
    }

    # Oracle positive control: only the target changes on final-decision rows.
    positive_reference = q0.copy()
    final = np.asarray([model.state(i).remaining == 1 for i in range(len(model.next_state))])
    logits = np.log(q0[final]) - 2 * target0[final] / TAU
    logits -= logits.max(axis=1, keepdims=True)
    positive_reference[final] = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    positive_optimal = independent_solution(model, positive_reference, "R0")
    positive_target = TAU * centered(positive_optimal["log_policy"])
    nonflat = final & (np.ptp(target0, axis=1) > 1e-12)
    positive = exact_metrics(model, actor0_lp, positive_reference, positive_optimal)
    positive.update({
        "target_mse": float(np.mean((target0[bank] - positive_target[bank]) ** 2)),
        "readout_mse": 0.0,
        "eta_max_abs": 0.0,
        "g_max_abs": float(np.max(np.abs(target0[bank] - positive_target[bank]))),
        "last_step_reversal_max_error": float(np.max(np.abs(positive_target[nonflat] + target0[nonflat]))) if np.any(nonflat) else 0.0,
        "nonflat_last_step_states": int(np.sum(nonflat)),
    })
    return rows, natural_rows, {
        "post_selection_rename_max_error": rename_error,
        "post_selection_metric_errors": rename_metric_errors,
        "wrong_column_alignment_max_error": wrong_error,
        "wrong_alignment_detected": bool(wrong_error > 1e-12),
    }, positive, state_arrays


def _cell(experiment, split, model_id, rendering):
    return Path(experiment) / "evaluations" / split / model_id / rendering


def analyze(experiment=EXPERIMENT):
    experiment = Path(experiment).resolve()
    read_config(experiment)
    results = {
        "status": "complete",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "experiment": str(experiment),
        "config_sha256": digest(experiment / "config.json"),
        "bootstrap": {"samples": BOOTSTRAP_SAMPLES, "seed": BOOTSTRAP_SEED, "unit": "paired task within family"},
        "splits": {},
        "repeat_controls": {},
        "source_tables": {},
    }
    for model_id in MODEL_SPECS:
        path = experiment / "controls/repeat" / model_id / "metrics.json"
        if not path.exists():
            raise RuntimeError(f"Missing repeat control: {model_id}")
        results["repeat_controls"][model_id] = read(path)
        results["source_tables"][str(path)] = digest(path)

    raw_rows, detail_rows = [], []
    for split in SPLIT_QUOTAS:
        models = load_models(experiment, split)
        for model_id in MODEL_SPECS:
            for rendering in RENDERINGS:
                completion = _cell(experiment, split, model_id, rendering) / "completion.json"
                if not completion.exists():
                    raise RuntimeError(f"Missing fixed evaluation cell: {split}/{model_id}/{rendering}")
                results["source_tables"][str(completion)] = digest(completion)

        per_task = []
        for model in models:
            base_tables = {rendering: policy_table(_cell(experiment, split, "base", rendering), model) for rendering in RENDERINGS}
            for model_id in MODEL_SPECS:
                actor_tables = {rendering: policy_table(_cell(experiment, split, model_id, rendering), model) for rendering in RENDERINGS}
                main, natural, controls, positive, state_arrays = analyze_task(model, actor_tables, base_tables)
                state_path = experiment / "analysis/state_arrays" / split / model_id / f"{model.task.key}.npz"
                state_path.parent.mkdir(parents=True, exist_ok=True)
                with state_path.open("wb") as stream:
                    np.savez_compressed(stream, **state_arrays)
                for rendering in RENDERINGS:
                    row = {
                        "split": split,
                        "model_id": model_id,
                        "rendering_id": rendering,
                        "task_seed": model.task.seed,
                        "family": model.task.family,
                        "content_sha256": model.task.fingerprint,
                        **main[rendering],
                    }
                    raw_rows.append(row)
                per_task.append({
                    "model_id": model_id,
                    "task_seed": model.task.seed,
                    "family": model.task.family,
                    "content_sha256": model.task.fingerprint,
                    "main": main,
                    "natural_reference": natural,
                    "rename_control": controls,
                    "target_positive_control": positive,
                    "state_arrays": str(state_path.relative_to(experiment)),
                    "state_arrays_sha256": digest(state_path),
                })
                detail_rows.append(per_task[-1] | {"split": split})

        summary = {"tasks": len(models), "models": {}, "controls": {}}
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        family_counts = {family: sum(model.task.family == family for model in models) for family in FAMILIES}
        shared_draws = {
            family: rng.integers(0, count, size=(BOOTSTRAP_SAMPLES, count))
            for family, count in family_counts.items()
        }
        for model_id in MODEL_SPECS:
            selected = [row for row in per_task if row["model_id"] == model_id]
            by_family = {family: [row for row in selected if row["family"] == family] for family in FAMILIES}

            def summarize(getter):
                values = {family: np.asarray([getter(row) for row in rows], dtype=float) for family, rows in by_family.items()}
                samples = np.mean([values[family][shared_draws[family]].mean(1) for family in FAMILIES], axis=0)
                low, high = np.quantile(samples, [.025, .975])
                return {
                    "mean": float(np.mean([value.mean() for value in values.values()])),
                    "lower": float(low),
                    "upper": float(high),
                    "per_family": {family: float(value.mean()) for family, value in values.items()},
                }

            model_summary = {"main": {}, "natural_reference": {}}
            metric_names = [
                "target_mse", "readout_mse", "semantic_tv", "argmax_change_rate",
                "regularized_return", "success_probability", "discounted_task_reward",
                "discounted_kl_penalty", "mismatch", "ceiling", "delta_tok_max",
            ]
            for rendering in RENDERINGS:
                model_summary["main"][rendering] = {}
                model_summary["natural_reference"][rendering] = {}
                for metric in metric_names:
                    model_summary["main"][rendering][metric] = summarize(lambda row, r=rendering, m=metric: row["main"][r][m])
                for metric in ("target_mse_vs_R0", "readout_mse_vs_R0", "regularized_return", "success_probability", "mismatch", "ceiling"):
                    model_summary["natural_reference"][rendering][metric] = summarize(lambda row, r=rendering, m=metric: row["natural_reference"][r][m])
                for metric in ("regularized_return", "success_probability", "mismatch"):
                    model_summary["main"][rendering][f"delta_{metric}_vs_R0"] = summarize(
                        lambda row, r=rendering, m=metric: row["main"][r][m] - row["main"]["R0"][m]
                    )
            model_summary["positive_control"] = {
                metric: summarize(lambda row, m=metric: row["target_positive_control"][m])
                for metric in ("target_mse", "readout_mse", "g_max_abs", "last_step_reversal_max_error", "regularized_return", "mismatch")
            }
            model_summary["validation"] = {
                "max_target_equivalence_error": max(row["main"][r]["ceiling_equivalence_error"] for row in selected for r in RENDERINGS),
                "max_optimal_policy_equivalence_error": max(row["main"][r]["optimal_policy_equivalence_max_error"] for row in selected for r in RENDERINGS),
                "max_gap_identity_error": max(row["main"][r]["gap_identity_error"] for row in selected for r in RENDERINGS),
                "max_return_decomposition_error": max(row["main"][r]["return_decomposition_error"] for row in selected for r in RENDERINGS),
                "max_deployment_identity_error": max(row["main"][r]["deployment_identity_max_error"] for row in selected for r in RENDERINGS),
                "max_pairwise_bound_violation": max(row["main"][r]["pairwise_bound_max_violation"] for row in selected for r in RENDERINGS),
                "max_l2_bound_violation": max(row["main"][r]["l2_bound_max_violation"] for row in selected for r in RENDERINGS),
                "max_rename_error": max(row["rename_control"]["post_selection_rename_max_error"] for row in selected),
                "max_rename_metric_error": max(value for row in selected for value in row["rename_control"]["post_selection_metric_errors"].values()),
                "wrong_alignment_detection_rate": float(np.mean([row["rename_control"]["wrong_alignment_detected"] for row in selected])),
                "max_positive_reversal_error": max(row["target_positive_control"]["last_step_reversal_max_error"] for row in selected),
                "max_delta_j_plus_delta_m_error": max(abs(
                    row["main"][r]["regularized_return"] - row["main"]["R0"]["regularized_return"]
                    + row["main"][r]["mismatch"] - row["main"]["R0"]["mismatch"]
                ) for row in selected for r in RENDERINGS),
            }
            summary["models"][model_id] = model_summary
        results["splits"][split] = summary

    output = experiment / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "readout_data.json", results)
    with (output / "task_rows.jsonl").open("w") as stream:
        for row in raw_rows:
            stream.write(json.dumps(row) + "\n")
    with (output / "task_details.jsonl").open("w") as stream:
        for row in detail_rows:
            stream.write(json.dumps(row) + "\n")
    write_json(output / "completion.json", {
        "status": "complete",
        "data_sha256": digest(output / "readout_data.json"),
        "rows_sha256": digest(output / "task_rows.jsonl"),
        "details_sha256": digest(output / "task_details.jsonl"),
        "test_rows": sum(row["split"] == "test" for row in raw_rows),
    })
    print(json.dumps({"analysis": str(output / 'readout_data.json'), "rows": len(raw_rows)}), flush=True)
    return results


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, default=EXPERIMENT)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    commands.add_parser("seal")
    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser.add_argument("--model", choices=MODEL_SPECS, required=True)
    evaluate_parser.add_argument("--rendering", choices=RENDERINGS, required=True)
    evaluate_parser.add_argument("--split", choices=SPLIT_QUOTAS, required=True)
    evaluate_parser.add_argument("--device", default="cuda:0")
    repeat_parser = commands.add_parser("repeat")
    repeat_parser.add_argument("--output", type=Path, required=True)
    repeat_parser.add_argument("--model", choices=MODEL_SPECS, required=True)
    repeat_parser.add_argument("--device", default="cuda:0")
    commands.add_parser("analyze")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        manifest = prepare_data(args.experiment)
        print(json.dumps({"tasks": len(manifest["entries"]), "suite": str(args.experiment / 'suite')}), flush=True)
    elif args.command == "seal":
        print(json.dumps({"config": str(args.experiment / 'config.json'), "sha256": digest(args.experiment / 'config.json') if (args.experiment / 'config.json').exists() else None, "sealed": bool(seal(args.experiment))}), flush=True)
    elif args.command == "evaluate":
        evaluate_cell(args.experiment, args.output, args.model, args.rendering, args.split, args.device)
    elif args.command == "repeat":
        repeat_forward(args.experiment, args.output, args.model, args.device)
    else:
        analyze(args.experiment)


if __name__ == "__main__":
    main()
