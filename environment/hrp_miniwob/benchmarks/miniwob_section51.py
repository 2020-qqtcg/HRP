"""Frozen probability measurements for the complete Section 5.1 experiment.

This module owns fresh IID data, immutable cell identities, graph/rendering
construction, GPU forward passes, and experiment sealing.  Finite-sample
estimators and prediction/scoring live in ``miniwob_section51_analysis`` so a
test predictor can be audited independently from the exact scorer.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import time
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from hrp_miniwob.adapters.miniwob_harnesses import (
    Graph as HarnessGraph,
)
from hrp_miniwob.adapters.miniwob_harnesses import (
    LanguagePolicy as HarnessLanguagePolicy,
)
from hrp_miniwob.adapters.miniwob_harnesses import (
    Task as HarnessTask,
)
from hrp_miniwob.adapters.miniwob_readout import ReadoutRenderer
from hrp_miniwob.adapters.miniwob_section51 import (
    NeutralReadoutRenderer,
    controlled_kernel,
    controlled_reachable_states,
)
from hrp_miniwob.adapters.miniwob_v2 import LanguagePolicy
from hrp_miniwob.adapters.native_miniwob import write_json
from hrp_miniwob.benchmarks.miniwob_meta_harness import digest, model_files
from hrp_miniwob.harnesses.miniwob_harnesses import Candidate
from hrp_miniwob.suite import reachable_states
from hrp_miniwob.v2 import FAMILIES
from hrp_miniwob.v2.browser import Browser, Task
from hrp_miniwob.v2.model import compile_task
from hrp_miniwob.v2.suite import is_ood, verify_success_path

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "environment/artifacts/v2/section51_diagnostics_2026-09-16"
DOCS = ROOT / "environment/docs/v2/section51_diagnostics_2026-09-16"
READOUT = ROOT / "environment/artifacts/v2/readout_equivalence_2026-09-15"
FIGURE3 = ROOT / "environment/artifacts/v2/figure3_h0_h3_exactpg_2026-09-14"
E1 = ROOT / "environment/artifacts/v2/qwen35_2b_h0_h7_exactpg_seed42_2026-09-11"
BASE_MODEL = Path("models/Qwen3.5-2B")
H0_ADAPTER = E1 / "runs/exactpg_H0/checkpoint_001024/actor/adapter"
H3_RUN = FIGURE3 / "runs/seed42_H3"
H3_ADAPTER = H3_RUN / "checkpoint_001024/actor/adapter"
H0_CANDIDATE = ROOT / "environment/hrp_miniwob/harnesses/miniwob_h0.py"
H3_CANDIDATE = ROOT / "environment/hrp_miniwob/harnesses/miniwob_h3.py"

GAMMA, TAU, HORIZON = .95, .1, 10
START_SEED = 51_100_000
SPLIT_QUOTAS = {"calibration": 24, "test": 192}
B_RENDERINGS = ("R0", "R1", "R3")
C_RENDERINGS = ("R0", "R1", "R3")
D_HARNESSES = ("H0", "H3")
D_NODES = (0, 32, 64, 128, 256, 512, 1024)
PHYSICAL_GPUS = (0, 1, 2, 3)


def read(path):
    return json.loads(Path(path).read_text())


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def source_paths():
    paths = [
        Path(__file__),
        ROOT / "environment/hrp_miniwob/adapters/miniwob_section51.py",
        ROOT / "environment/hrp_miniwob/benchmarks/miniwob_section51_analysis.py",
        ROOT / "environment/hrp_miniwob/validation/check_miniwob_section51.py",
        ROOT / "environment/hrp_miniwob/orchestration/run_miniwob_section51.py",


        ROOT / "environment/hrp_miniwob/adapters/miniwob_readout.py",
        ROOT / "environment/hrp_miniwob/adapters/miniwob_harnesses.py",
        ROOT / "environment/hrp_miniwob/harnesses/miniwob_harnesses.py",
        ROOT / "environment/hrp_miniwob/v2/model.py",
        ROOT / "environment/hrp_miniwob/v2/browser.py",
        ROOT / "environment/hrp_miniwob/v2/suite.py",
        ROOT / "environment/hrp_miniwob/oracle.py",
        ROOT / "environment/training/uv.lock",
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Section 5.1 source set is incomplete: " + ", ".join(missing))
    return paths


def source_hashes():
    return {str(path.relative_to(ROOT)): digest(path) for path in source_paths()}


def adapter_for_node(update: int) -> Path:
    if update == 0:
        return H0_ADAPTER
    if update not in D_NODES:
        raise ValueError(f"Unknown adaptation node: {update}")
    return H3_RUN / f"checkpoint_{update:06d}/actor/adapter"


def all_model_paths():
    paths = {"base": BASE_MODEL, "h0_final": H0_ADAPTER, "h3_adapt_final": H3_ADAPTER}
    paths.update({f"h3_update_{update:04d}": adapter_for_node(update) for update in D_NODES})
    return paths


def model_hashes():
    return {key: model_files(path) for key, path in all_model_paths().items()}


def proposal():
    return {
        "schema_version": 1,
        "created_for": "Chapter 5.1 finite-measurement and outcome-prediction experiment",
        "families": list(FAMILIES),
        "splits_per_family": SPLIT_QUOTAS,
        "start_seed": START_SEED,
        "family_seed_offset": 1_000_000,
        "horizon": HORIZON,
        "gamma": GAMMA,
        "tau": TAU,
        "training": "none; all checkpoints frozen; source and adaptation training seed 42",
        "physical_gpus": list(PHYSICAL_GPUS),
        "probability_protocols": {
            "B": "H0 shared-reference readout prompts R0/R1/R3",
            "C": "execution-neutral H0 prompts R0/R1/R3 shared across all p/c settings",
            "D": "native Figure 3 H0/H3 prompts and per-interface frozen base reference",
        },
        "finite_sampling": {
            "m": [4, 16, 64],
            "N": [16, 64, 256],
            "alpha_main": .5,
            "alpha_appendix": [.1, .5, 1.0],
            "calibration_repeats": 100,
            "nested_prefix": True,
        },
        "execution_sweeps": {
            "baseline": {"reliability": .8, "click_cost": .01},
            "reliability": [.5, .6, .7, .75, .8, .85, .9, .95, 1.0],
            "cost": [0, .0025, .005, .0075, .01, .015, .02, .03, .05],
            "calibration_reliability": [.5, .8, 1.0],
            "calibration_cost": [0, .01, .05],
            "renderings": list(C_RENDERINGS),
            "rollouts_per_task": 32,
        },
        "adaptation_nodes": list(D_NODES),
        "bootstrap": {"samples": 2000, "seed": 5205, "unit": "paired native task within family"},
        "random_streams": {
            "B_sampling": 5101,
            "C_probe": 5102,
            "C_rollout": 5103,
            "D_probe": 5104,
            "D_rollout": 5105,
            "bootstrap": 5205,
        },
        "test_order": "save and hash predictions before exact or rollout scoring",
    }


def collect_exclusions(experiment):
    contents, seeds, manifests = set(), set(), {}
    for path in sorted((ROOT / "environment/artifacts").rglob("manifest.json")):
        if experiment in path.parents or any(
            part in {"source_snapshot", "runtime_source", "source", "vendor"} for part in path.parts
        ):
            continue
        try:
            record = read(path)
        except (json.JSONDecodeError, OSError):
            continue
        used = False
        if record.get("version") == "v2" and isinstance(record.get("entries"), list):
            for entry in record["entries"]:
                if "content_sha256" in entry and "seed" in entry:
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
                    native = Task(
                        payload["seed"],
                        "click-checkboxes",
                        payload["instruction"],
                        {"labels": payload["labels"]},
                        0,
                    )
                    contents.add(native.fingerprint)
                    seeds.add(int(payload["seed"]))
                    used = True
        if used:
            manifests[str(path)] = digest(path)
    return {"manifest_sha256": manifests, "content_sha256": sorted(contents), "seeds": sorted(seeds)}


def prepare_data(experiment=EXPERIMENT):
    experiment = Path(experiment).resolve()
    experiment.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    proposal_path = experiment / "proposal.json"
    if proposal_path.exists():
        if read(proposal_path) != proposal():
            raise ValueError("Existing Section 5.1 proposal differs from the fixed design")
    else:
        write_json(proposal_path, proposal())
    manifest_path = experiment / "suite/manifest.json"
    if manifest_path.exists():
        return read(manifest_path)

    exclusion_path = experiment / "exclusions.json"
    exclusions = read(exclusion_path) if exclusion_path.exists() else collect_exclusions(experiment)
    if not exclusion_path.exists():
        write_json(exclusion_path, exclusions)
    for path, expected in exclusions["manifest_sha256"].items():
        if digest(path) != expected:
            raise ValueError(f"An exclusion manifest changed during preparation: {path}")
    forbidden_contents = set(exclusions["content_sha256"])
    forbidden_seeds = set(exclusions["seeds"])

    suite = experiment / "suite"
    (suite / "tasks").mkdir(parents=True, exist_ok=True)
    journal = experiment / "task_journal.jsonl"
    entries = [json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []
    seen = {entry["content_sha256"] for entry in entries}
    if len(seen) != len(entries) or seen & forbidden_contents:
        raise AssertionError("Resumed task journal violates exclusion or uniqueness rules")

    with Browser(horizon=HORIZON) as browser:
        for family_index, family in enumerate(FAMILIES):
            counts = Counter(entry["split"] for entry in entries if entry["family"] == family)
            previous = [entry["seed"] for entry in entries if entry["family"] == family]
            seed = max(previous) + 1 if previous else START_SEED + family_index * 1_000_000
            scanned = 0
            while any(counts[split] < quota for split, quota in SPLIT_QUOTAS.items()):
                if scanned >= 100_000:
                    raise RuntimeError(f"Fresh IID scan exhausted for {family}")
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
                    "native_reachable_states": len(reachable_states(model)),
                    "controlled_reachable_states": len(
                        controlled_reachable_states(controlled_kernel(model, .5, .01))
                    ),
                }
                with journal.open("a") as stream:
                    stream.write(json.dumps(entry) + "\n")
                    stream.flush()
                entries.append(entry)
                seen.add(task.fingerprint)
                counts[split] += 1
                write_json(
                    experiment / "preparation.json",
                    {"status": "running", "family": family, "counts": dict(counts), "tasks": len(entries)},
                )
            print(json.dumps({"family": family, "counts": dict(counts)}), flush=True)

    expected = len(FAMILIES) * sum(SPLIT_QUOTAS.values())
    if len(entries) != expected or len(seen) != expected:
        raise AssertionError("Prepared suite has the wrong number of unique tasks")
    manifest = {
        "version": "v2",
        "status": "complete",
        "profile": "section51_diagnostics",
        "horizon": HORIZON,
        "families": list(FAMILIES),
        "entries": entries,
        "exclusions_sha256": digest(exclusion_path),
        "validation": "live native success replay; all retained manifests excluded by seed and public content",
    }
    write_json(manifest_path, manifest)
    splits = {split: [entry["seed"] for entry in entries if entry["split"] == split] for split in SPLIT_QUOTAS}
    write_json(experiment / "splits.json", splits)
    write_json(
        experiment / "preparation.json",
        {
            "status": "complete",
            "counts": {split: len(values) for split, values in splits.items()},
            "content_overlap": 0,
            "seed_overlap": 0,
            "excluded_manifests": len(exclusions["manifest_sha256"]),
        },
    )
    return manifest


def load_models(experiment, split):
    experiment = Path(experiment)
    manifest = read(experiment / "suite/manifest.json")
    if manifest.get("status") != "complete" or manifest.get("version") != "v2":
        raise ValueError("Section 5.1 suite is incomplete")
    models = []
    for entry in manifest["entries"]:
        if entry["split"] != split:
            continue
        path = experiment / "suite" / entry["json"]
        if digest(path) != entry["sha256"]:
            raise ValueError(f"Prepared task changed: {path}")
        task = Task(**read(path))
        if task.fingerprint != entry["content_sha256"]:
            raise ValueError("Task content fingerprint changed")
        models.append(compile_task(task, HORIZON))
    expected = len(FAMILIES) * SPLIT_QUOTAS[split]
    if len(models) != expected:
        raise ValueError(f"Expected {expected} {split} tasks, found {len(models)}")
    return models


def candidate_path(harness):
    return {"H0": H0_CANDIDATE, "H3": H3_CANDIDATE}[harness]


def harness_graph(model, harness):
    """Compile current H0/H3 without loading actor or reference weights."""

    candidate = Candidate(candidate_path(harness))
    if candidate.profile != harness:
        raise AssertionError("Harness candidate profile disagrees with its current identifier")
    task = HarnessTask(**asdict(model.task), candidate=candidate)
    shape = HORIZON * task.memories * task.physical_states, len(task.actions)
    graph = HarnessGraph(
        task,
        HORIZON,
        candidate,
        np.full(shape, -1, dtype=np.int64),
        np.zeros(shape, dtype=np.float64),
    )
    for index in range(shape[0]):
        state = graph.state(index)
        for column, action in enumerate(task.actions):
            memory, primitives, _ = candidate.request(
                task,
                state.memory,
                action,
                lambda *_: (_ for _ in ()).throw(AssertionError("H0/H3 must not invoke a reviewer")),
            )
            physical, reward = state.physical, 0.0
            for primitive in primitives:
                native_column = task.native_actions.index(primitive)
                reward = float(model.physical_rewards[physical, native_column])
                physical = int(model.physical_next[physical, native_column])
                if physical < 0:
                    break
            graph.task_rewards[index, column] = reward
            if physical >= 0 and state.remaining > 1:
                graph.next_state[index, column] = graph.index(state.remaining - 1, memory, physical)
    return graph


def cell_specs(split):
    if split not in SPLIT_QUOTAS:
        raise ValueError(f"Unknown split: {split}")
    specs = {}
    specs["B_base_R0"] = {
        "study": "B", "split": split, "model_id": "base", "adapter": None, "rendering": "R0"
    }
    for rendering in B_RENDERINGS:
        specs[f"B_h0_final_{rendering}"] = {
            "study": "B", "split": split, "model_id": "h0_final", "adapter": H0_ADAPTER,
            "rendering": rendering,
        }
    specs["C_base_R0"] = {
        "study": "C", "split": split, "model_id": "base", "adapter": None, "rendering": "R0"
    }
    for model_id, adapter in (("h0_final", H0_ADAPTER), ("h3_adapt_final", H3_ADAPTER)):
        for rendering in C_RENDERINGS:
            specs[f"C_{model_id}_{rendering}"] = {
                "study": "C", "split": split, "model_id": model_id, "adapter": adapter,
                "rendering": rendering,
            }
    for harness in D_HARNESSES:
        specs[f"D_base_{harness}"] = {
            "study": "D", "split": split, "model_id": "base", "adapter": None, "harness": harness,
        }
    for update in D_NODES:
        adapter = adapter_for_node(update)
        for harness in D_HARNESSES:
            specs[f"D_h3_update_{update:04d}_{harness}"] = {
                "study": "D", "split": split, "model_id": f"h3_update_{update:04d}",
                "adapter": adapter, "harness": harness, "update": update,
            }
    return specs


def cell_directory(experiment, split, cell_id):
    return Path(experiment) / "probabilities" / split / cell_id


def cell_complete(experiment, split, cell_id):
    output = cell_directory(experiment, split, cell_id)
    completion = output / "completion.json"
    return completion.exists() and read(completion).get("accounting_sha256") == digest(output / "accounting.json")


def graph_and_prompts(model, spec):
    if spec["study"] in {"B", "C"}:
        graph = model
        renderer_class = ReadoutRenderer if spec["study"] == "B" else NeutralReadoutRenderer
        renderer = renderer_class(spec["rendering"])
        if spec["study"] == "B":
            bank = reachable_states(model)
        else:
            bank = controlled_reachable_states(controlled_kernel(model, .5, .01))
        rendered = [renderer.render(model.state(int(index))) for index in bank]
        return graph, bank, rendered
    graph = harness_graph(model, spec["harness"])
    bank = reachable_states(graph)
    rendered = []
    for index in bank:
        messages, codes = graph.candidate.messages(graph.state(int(index)))
        payload = json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()
        rendered.append(SimpleNamespace(
            messages=messages,
            semantic_codes=codes,
            prompt_sha256=sha256_bytes(payload),
            mapping={"harness": spec["harness"], "semantic_order": list(graph.task.actions)},
        ))
    return graph, bank, rendered


def load_policy(spec, device):
    policy_class = HarnessLanguagePolicy if spec["study"] == "D" else LanguagePolicy
    policy = policy_class(
        str(BASE_MODEL),
        seed=42,
        device=device,
        adapter_path=str(spec["adapter"]) if spec["adapter"] else None,
        trainable=False,
        rank=16,
    )
    if any(parameter.requires_grad for parameter in policy.model.parameters()):
        raise AssertionError("Section 5.1 probability measurement loaded trainable weights")
    policy.model.eval()
    return policy


def task_table_path(output, model):
    return Path(output) / "tasks" / f"{model.task.key}.npz"


def measure_task(policy, model, spec, output):
    path = task_table_path(output, model)
    metadata_path = path.with_suffix(".json")
    if path.exists() and metadata_path.exists():
        metadata = read(metadata_path)
        if metadata["content_sha256"] != model.task.fingerprint or metadata["table_sha256"] != digest(path):
            raise ValueError("Resumed task table identity or checksum changed")
        return metadata
    path.parent.mkdir(parents=True, exist_ok=True)
    graph, bank, rendered = graph_and_prompts(model, spec)
    prompts = [row.messages for row in rendered]
    codes = [row.semantic_codes for row in rendered]
    before = (policy.forward_observations, policy.forward_prompt_tokens, policy.forward_microbatches)
    started = time.monotonic()
    with torch.no_grad():
        log_policy = policy.batch_log_probabilities(prompts, codes).detach().cpu().numpy()
    if not np.all(np.isfinite(log_policy)) or not np.allclose(
        np.exp(log_policy).sum(1), 1, atol=1e-12, rtol=0
    ):
        raise AssertionError("Actor did not return a complete normalized menu distribution")
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            bank=bank,
            log_policy=log_policy,
            prompt_sha256=np.asarray([row.prompt_sha256 for row in rendered]),
            semantic_actions=np.asarray(graph.task.actions),
            semantic_codes=np.asarray(codes[0]),
            next_state=graph.next_state,
            task_rewards=graph.task_rewards,
            initial_state=np.asarray(graph.initial_state),
        )
    temporary.replace(path)
    after = (policy.forward_observations, policy.forward_prompt_tokens, policy.forward_microbatches)
    metadata = {
        "task_seed": model.task.seed,
        "family": model.task.family,
        "content_sha256": model.task.fingerprint,
        "study": spec["study"],
        "rendering": spec.get("rendering"),
        "harness": spec.get("harness"),
        "states": len(bank),
        "graph_states": len(graph.next_state),
        "actions": len(graph.task.actions),
        "mapping": rendered[0].mapping,
        "wall_seconds": time.monotonic() - started,
        "forward_state_rows": after[0] - before[0],
        "prompt_tokens": after[1] - before[1],
        "forward_microbatches": after[2] - before[2],
        "table_sha256": digest(path),
    }
    write_json(metadata_path, metadata)
    return metadata


def _model_hash_key(spec):
    if spec["model_id"] == "base":
        return "base"
    return spec["model_id"]


def measure_cell(experiment, split, cell_id, device):
    experiment = Path(experiment).resolve()
    specs = cell_specs(split)
    if cell_id not in specs:
        raise ValueError(f"Unknown fixed probability cell: {split}/{cell_id}")
    spec = specs[cell_id]
    output = cell_directory(experiment, split, cell_id)
    output.mkdir(parents=True, exist_ok=True)
    measured_path = BASE_MODEL if spec["model_id"] == "base" else spec["adapter"]
    identity = {
        "proposal_sha256": digest(experiment / "proposal.json"),
        "suite_manifest_sha256": digest(experiment / "suite/manifest.json"),
        "split": split,
        "cell_id": cell_id,
        "study": spec["study"],
        "model_id": spec["model_id"],
        "model_files_sha256": model_files(measured_path),
        "adapter": str(spec["adapter"]) if spec["adapter"] else None,
        "rendering": spec.get("rendering"),
        "harness": spec.get("harness"),
        "update": spec.get("update"),
        "device": device,
    }
    identity_path = output / "identity.json"
    if identity_path.exists() and read(identity_path) != identity:
        raise ValueError("Probability directory belongs to another immutable cell")
    if not identity_path.exists():
        write_json(identity_path, identity)
    if cell_complete(experiment, split, cell_id):
        print(json.dumps(read(output / "completion.json")), flush=True)
        return

    policy = load_policy(spec, device)
    rows, started = [], time.monotonic()
    try:
        models = load_models(experiment, split)
        for number, model in enumerate(models, 1):
            rows.append(measure_task(policy, model, spec, output))
            write_json(
                output / "progress.json",
                {
                    "status": "running",
                    "completed_tasks": number,
                    "total_tasks": len(models),
                    "updated_at_utc": datetime.now(UTC).isoformat(),
                },
            )
        accounting = {
            "status": "complete",
            "split": split,
            "cell_id": cell_id,
            "study": spec["study"],
            "tasks": len(rows),
            "states": sum(row["states"] for row in rows),
            "forward_state_rows": sum(row["forward_state_rows"] for row in rows),
            "prompt_tokens": sum(row["prompt_tokens"] for row in rows),
            "forward_microbatches": sum(row["forward_microbatches"] for row in rows),
            "wall_seconds": time.monotonic() - started,
            "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "model_metadata": policy.metadata,
        }
        write_json(output / "accounting.json", accounting)
        completion = {
            "status": "complete",
            "identity_sha256": digest(identity_path),
            "accounting_sha256": digest(output / "accounting.json"),
            "task_table_hashes_sha256": sha256_bytes(
                json.dumps({row["task_seed"]: row["table_sha256"] for row in rows}, sort_keys=True).encode()
            ),
            "completed_at_utc": datetime.now(UTC).isoformat(),
        }
        write_json(output / "completion.json", completion)
        print(json.dumps(completion), flush=True)
    finally:
        del policy
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_table(experiment, split, cell_id, model):
    path = task_table_path(cell_directory(experiment, split, cell_id), model)
    metadata = read(path.with_suffix(".json"))
    if digest(path) != metadata["table_sha256"] or metadata["content_sha256"] != model.task.fingerprint:
        raise ValueError(f"Probability table failed checksum/identity validation: {path}")
    with np.load(path) as saved:
        return {key: saved[key].copy() for key in saved.files}


def audit_probability_tables(experiment, split):
    """Rebuild graph/rendering metadata and compare every saved table."""

    checks = states = prompts = 0
    models = load_models(experiment, split)
    for cell_id, spec in cell_specs(split).items():
        if not cell_complete(experiment, split, cell_id):
            raise RuntimeError(f"Incomplete probability cell during audit: {split}/{cell_id}")
        for model in models:
            table = load_table(experiment, split, cell_id, model)
            graph, bank, rendered = graph_and_prompts(model, spec)
            if not np.array_equal(table["bank"], bank):
                raise AssertionError("Saved probability bank differs from current reachable-state construction")
            if not np.array_equal(table["next_state"], graph.next_state):
                raise AssertionError("Saved probability graph differs from current transition construction")
            if not np.array_equal(table["task_rewards"], graph.task_rewards):
                raise AssertionError("Saved probability rewards differ from current graph")
            if int(table["initial_state"]) != graph.initial_state:
                raise AssertionError("Saved probability table has the wrong initial state")
            if tuple(str(value) for value in table["semantic_actions"]) != tuple(graph.task.actions):
                raise AssertionError("Saved semantic action columns changed")
            if tuple(str(value) for value in table["semantic_codes"]) != tuple(rendered[0].semantic_codes):
                raise AssertionError("Saved one-token decoder changed")
            current_hashes = np.asarray([row.prompt_sha256 for row in rendered])
            if not np.array_equal(table["prompt_sha256"], current_hashes):
                raise AssertionError("Saved prompts differ from the current renderer")
            checks += 1
            states += len(graph.next_state)
            prompts += len(bank)
    return {
        "status": "passed",
        "split": split,
        "cells": len(cell_specs(split)),
        "tasks_per_cell": len(models),
        "task_cell_checks": checks,
        "graph_state_rows_checked": states,
        "prompt_rows_checked": prompts,
        "scope": "all saved graphs, rewards, state banks, semantic columns, decoder codes, and prompt hashes rebuilt with sealed source",
    }


def full_log_policy(table):
    shape = table["next_state"].shape
    actions = shape[1]
    result = np.full(shape, -np.log(actions), dtype=np.float64)
    result[table["bank"]] = table["log_policy"]
    return result


def audit_readout_source():
    completion = read(READOUT / "analysis/completion.json")
    scheduler = read(READOUT / "scheduler.json")
    if completion.get("status") != "complete" or scheduler.get("status") != "complete":
        raise RuntimeError("The completed readout experiment failed its completion audit")
    required = [
        READOUT / "config.json",
        READOUT / "analysis/readout_data.json",
        ROOT / "environment/docs/v2/readout_equivalence_2026-09-15/REPORT.md",


    ]
    return {
        "status": "passed",
        "completed_cells": 36,
        "scheduler": {"calibration_jobs": scheduler["calibration_jobs"], "test_jobs": scheduler["test_jobs"]},
        "files_sha256": {str(path): digest(path) for path in required},
        "source_config_sha256": digest(READOUT / "config.json"),
        "training_seed_statement": "H0 source and H3 adaptation use seed 42; three model conditions are not seeds",
    }


def seal(experiment=EXPERIMENT):
    experiment = Path(experiment).resolve()
    if read(experiment / "preparation.json").get("status") != "complete":
        raise RuntimeError("Data preparation is incomplete")
    for name in ("cpu.json", "gpu.json"):
        if read(experiment / "acceptance" / name).get("status") != "passed":
            raise RuntimeError(f"Acceptance did not pass: {name}")
    calibration = read(experiment / "calibration/frozen_choices.json")
    if calibration.get("status") != "frozen":
        raise RuntimeError("Calibration choices must be frozen before sealing")
    if any(not cell_complete(experiment, "calibration", key) for key in cell_specs("calibration")):
        raise RuntimeError("All calibration probability cells must complete before sealing")
    config = proposal() | {
        "sealed_at_utc": datetime.now(UTC).isoformat(),
        "suite_manifest_sha256": digest(experiment / "suite/manifest.json"),
        "splits_sha256": digest(experiment / "splits.json"),
        "exclusions_sha256": digest(experiment / "exclusions.json"),
        "model_files_sha256": model_hashes(),
        "source_sha256": source_hashes(),
        "calibration_choices_sha256": digest(experiment / "calibration/frozen_choices.json"),
        "readout_source_audit": audit_readout_source(),
        "harness_numbering": read(E1 / "harness_numbering.json"),
        "harness_numbering_sha256": digest(E1 / "harness_numbering.json"),
        "acceptance_sha256": {
            "cpu": digest(experiment / "acceptance/cpu.json"),
            "gpu": digest(experiment / "acceptance/gpu.json"),
        },
        "calibration_probability_audit": audit_probability_tables(experiment, "calibration"),
    }
    path = experiment / "config.json"
    if path.exists():
        old = read(path)
        config["sealed_at_utc"] = old["sealed_at_utc"]
        if old != config:
            raise ValueError("Sealed Section 5.1 configuration changed")
        return old
    snapshot = experiment / "source_snapshot"
    for source in source_paths():
        target = snapshot / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    write_json(path, config)
    write_json(
        experiment / "manifest.json",
        {
            "status": "sealed",
            "config_sha256": digest(path),
            "primary_split": "test",
            "test_blinding": "predictor outputs are sealed before exact or rollout score outputs",
            "training_seed_statement": "all checkpoint training/adaptation used only seed 42",
        },
    )
    return config


def verify_seal(experiment=EXPERIMENT):
    experiment = Path(experiment)
    config = read(experiment / "config.json")
    checks = {
        "suite/manifest.json": config["suite_manifest_sha256"],
        "splits.json": config["splits_sha256"],
        "exclusions.json": config["exclusions_sha256"],
        "calibration/frozen_choices.json": config["calibration_choices_sha256"],
    }
    for relative, expected in checks.items():
        if digest(experiment / relative) != expected:
            raise ValueError(f"Sealed input changed: {relative}")
    if source_hashes() != config["source_sha256"]:
        raise ValueError("Section 5.1 source differs from the sealed snapshot")
    if model_hashes() != config["model_files_sha256"]:
        raise ValueError("A frozen model/checkpoint changed after sealing")
    if digest(E1 / "harness_numbering.json") != config["harness_numbering_sha256"]:
        raise ValueError("Current H0--H5 numbering record changed after sealing")
    return config


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare-data")
    prepare.add_argument("--experiment", type=Path, default=EXPERIMENT)
    measure = sub.add_parser("measure-cell")
    measure.add_argument("--experiment", type=Path, default=EXPERIMENT)
    measure.add_argument("--split", choices=tuple(SPLIT_QUOTAS), required=True)
    measure.add_argument("--cell", required=True)
    measure.add_argument("--device", default="cuda:0")
    seal_parser = sub.add_parser("seal")
    seal_parser.add_argument("--experiment", type=Path, default=EXPERIMENT)
    list_parser = sub.add_parser("list-cells")
    list_parser.add_argument("--split", choices=tuple(SPLIT_QUOTAS), required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    torch.set_num_threads(2)
    if args.command == "prepare-data":
        result = prepare_data(args.experiment)
        print(json.dumps({"status": result["status"], "tasks": len(result["entries"])}), flush=True)
    elif args.command == "measure-cell":
        measure_cell(args.experiment, args.split, args.cell, args.device)
    elif args.command == "seal":
        result = seal(args.experiment)
        print(json.dumps({"status": "sealed", "sha256": digest(Path(args.experiment) / "config.json")}), flush=True)
    else:
        print(json.dumps({key: {k: str(v) if isinstance(v, Path) else v for k, v in spec.items()}
                          for key, spec in cell_specs(args.split).items()}, indent=2))


if __name__ == "__main__":
    main()
