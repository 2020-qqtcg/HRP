"""Section 5.1: exact H0--H5 selection with frozen Qwen3.5-9B as its own prior."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from hrp_miniwob.finite_horizon import (
    independent_success_rollouts,
    self_reference_metrics,
)

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "environment/artifacts/v2/section51_qwen35_9b_self_reference_2026-09-19"
DOCS = ROOT / "environment/docs/v2/section51_qwen35_9b_self_reference_2026-09-19"
MODEL = Path("models/Qwen3.5-9B")
GRAPH_SOURCE = ROOT / "environment/artifacts/v2/figure4_h0_h5_prediction_2026-09-17"
SUITE_SOURCES = {
    "calibration": ROOT / "environment/artifacts/v2/section51_diagnostics_2026-09-16",
    "test": ROOT / "environment/artifacts/v2/section51_measurement_v2_2026-09-16",
}
HARNESSES = tuple(f"H{i}" for i in range(6))
FAMILIES = ("click-checkboxes", "click-checkboxes-soft", "ascending-numbers")
GPUS = (4, 5, 6, 7)
HORIZON, TAU, ROLLOUTS = 10, .1, 256
OLD_OBJECTIVE = ("Optimize discounted reward gamma=0.95 and KL regularization "
                 "tau=0.1 to the independent frozen reference. ")
NEW_OBJECTIVE = "Maximize the probability of exact task completion within the decision budget. "
GRAPH_KEYS = ("bank", "semantic_actions", "semantic_codes", "next_state",
              "task_rewards", "initial_state")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def now():
    return datetime.now(UTC).isoformat()


def gpu_inventory():
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used",
        "--format=csv,noheader,nounits",
    ], text=True)
    devices = {}
    for line in output.splitlines():
        index, uuid, name, total, used = (item.strip() for item in line.split(","))
        if int(index) in GPUS:
            devices[index] = {"uuid": uuid, "name": name, "total_mib": int(total),
                              "used_mib_at_prepare": int(used)}
    if set(map(int, devices)) != set(GPUS):
        raise RuntimeError("The requested physical GPUs 4--7 are not all available")
    return devices


def prepare(experiment=EXPERIMENT):
    """Freeze task/graph inputs, model identity and executable source before inference."""
    experiment = Path(experiment).resolve()
    if (experiment / "manifest.json").exists():
        verify_sources(experiment)
        return read(experiment / "manifest.json")
    experiment.mkdir(parents=True, exist_ok=True)
    print("Preparing task graphs and model fingerprints", flush=True)
    entries, source_manifests = [], {}
    for split, source in SUITE_SOURCES.items():
        source_manifest = read(source / "suite/manifest.json")
        if source_manifest.get("status") != "complete" or source_manifest["horizon"] != HORIZON:
            raise ValueError("Source task suite is incomplete or has a different horizon")
        source_manifests[split] = {"path": str(source / "suite/manifest.json"),
                                   "sha256": digest(source / "suite/manifest.json")}
        rows = [row for row in source_manifest["entries"] if row["split"] == split]
        if split == "calibration":
            # One predetermined task per family, used only for hardware/implementation checks.
            rows = [next(row for row in rows if row["family"] == f) for f in FAMILIES]
        expected = 3 if split == "calibration" else 576
        if len(rows) != expected:
            raise ValueError(f"Unexpected {split} task count")
        for row in rows:
            task_file = source / "suite" / row["json"]
            if digest(task_file) != row["sha256"] or row["verified_success_steps"] > HORIZON:
                raise ValueError("Native task checksum or success replay changed")
            task = read(task_file)
            key = f"{row['family']}_{row['seed']}"
            target = experiment / "inputs/tasks" / f"{key}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(task_file, target)
            entry = {"key": key, "split": split, "family": row["family"],
                     "seed": row["seed"], "content_sha256": row["content_sha256"],
                     "task_path": str(target.relative_to(experiment)),
                     "task_sha256": digest(target), "graphs": {}}
            if task["family"] != row["family"] or task["seed"] != row["seed"]:
                raise AssertionError("Task identity differs from its manifest")
            for harness in HARNESSES:
                source_table = GRAPH_SOURCE / "probabilities" / split / harness / "tasks" / f"{key}.npz"
                meta = read(source_table.with_suffix(".json"))
                if (digest(source_table) != meta["table_sha256"]
                        or meta["content_sha256"] != entry["content_sha256"]):
                    raise ValueError("Existing harness graph failed provenance checks")
                target = experiment / "inputs/graphs" / split / harness / f"{key}.npz"
                target.parent.mkdir(parents=True, exist_ok=True)
                with np.load(source_table) as saved:
                    # Deliberately import no old actor/reference probabilities or metrics.
                    np.savez_compressed(target, **{k: saved[k] for k in GRAPH_KEYS})
                entry["graphs"][harness] = {
                    "path": str(target.relative_to(experiment)), "sha256": digest(target),
                    "source_table_sha256": meta["table_sha256"], "states": meta["states"],
                }
            entries.append(entry)
    test = [e for e in entries if e["split"] == "test"]
    if Counter(e["family"] for e in test) != dict.fromkeys(FAMILIES, 192):
        raise AssertionError("Test families must have equal, complete counts")
    if len({e["content_sha256"] for e in entries}) != len(entries):
        raise AssertionError("Calibration and test tasks must be content-disjoint")

    snapshot = experiment / "source_snapshot"
    sources = [*(ROOT / "environment/hrp_miniwob").rglob("*.py"),
               *(ROOT / "environment/hrp_miniwob").rglob("*.json"),
               *(ROOT / "src/hrp").glob("*.py"),


               ROOT / "environment/training/pyproject.toml", ROOT / "environment/training/uv.lock"]
    source_hashes = {}
    for source in sorted(sources):
        rel = source.relative_to(ROOT)
        target = snapshot / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        source_hashes[str(rel)] = digest(target)
    weights = sorted(p for p in MODEL.iterdir() if p.suffix in (".json", ".safetensors", ".jinja", ".txt"))
    if len([p for p in weights if p.suffix == ".safetensors"]) != 4:
        raise ValueError("Expected the complete local Qwen3.5-9B checkpoint")
    model_files = {p.name: {"sha256": digest(p), "size": p.stat().st_size,
                           "mtime_ns": p.stat().st_mtime_ns} for p in weights}
    manifest = {
        "schema_version": 1, "task_id": experiment.name, "created_at_utc": now(),
        "status": "prepared", "harnesses": list(HARNESSES), "families": list(FAMILIES),
        "model": str(MODEL), "model_files": model_files, "model_seed": 42,
        "training": "none; no project adapter; frozen off-the-shelf Qwen3.5-9B",
        "reference": "same frozen semantic policy as actor separately under each harness",
        "H4_reviewer": "retain archived independent frozen Qwen3.5-2B reviewer decisions in execution graph",
        "horizon": HORIZON, "tau": TAU, "gamma": 1., "explicit_cost": 0.,
        "objective": "finite-horizon undiscounted terminal binary success with self-reference KL",
        "occupancy": "unnormalized finite visit counts; no 1/(1-gamma)",
        "prompt_objective": {"replaced": OLD_OBJECTIVE, "replacement": NEW_OBJECTIVE},
        "decoder": {"temperature": 1., "thinking": False, "top_p": 1.,
                    "kind": "full legal single-token semantic menu", "dtype": "BF16",
                    "normalization": "FP64", "microbatch": 8},
        "test_tasks": 576, "preflight_tasks": 3, "rollouts_per_task_harness": ROLLOUTS,
        "rollout_scope": "independent samples on the native-audited compiled graph",
        "bootstrap_repeats": 2000, "bootstrap_seed": 51942,
        "statistics": "paired tasks, equal family weights; no training-seed variance",
        "selection_scores": ["C", "-M", "C-M"],
        "ranking_scope": "within the same fixed task distribution; oracle analysis, not sample-efficient selection",
        "deterministic_binary_identity": "For each individual task, C=tau*log(1+S*(exp(1/tau)-1)); C is monotone in success S. Aggregated C averages this nonlinear transform across tasks.",
        "physical_gpus": list(GPUS), "gpu_inventory": gpu_inventory(), "worker_shards": 4,
        "source_manifests": source_manifests, "graph_source": str(GRAPH_SOURCE),
        "graph_source_config_sha256": digest(GRAPH_SOURCE / "config.json"),
        "source_hashes": source_hashes, "entries": entries,
    }
    write(experiment / "manifest.json", manifest)
    write(experiment / "progress.json", {"status": "prepared", "total_evaluations": 3456})
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "REPORT.md").write_text(
        "# Section 5.1: Qwen3.5-9B self-reference\n\n"
        "Status: prepared. No deployment results yet.\n\n"
        "Frozen Qwen3.5-9B, no project post-training, H0–H5, 576 paired native-audited tasks "
        "(192 per family), ten actor decisions, undiscounted binary success, zero explicit cost, tau=0.1. "
        "The actor is its own harness-specific reference. H4 retains its archived independent "
        "Qwen3.5-2B reviewer. Only the objective wording in the actor prompt changes.\n\n"
        "Exact C, M, J and success use finite-horizon planning, visitation KL, forward reward "
        "accumulation and backward success evaluation. An additional 256 independent graph rollouts "
        "per task/harness measure sampled success. These are compiled-graph rollouts, not new browser runs. "
        "C-M=success is an identity in this protocol. No conclusion about baseline rankings is assumed.\n\n"
        "For a deterministic task with self-reference and binary terminal reward, "
        "C=tau*log(1+S*(exp(1/tau)-1)). Thus task-level C is already monotone in success. "
        "Harness-level C averages this nonlinear transform across the shared task set.\n\n"
        f"Manifest: `{experiment / 'manifest.json'}`\n"
        f"Progress: `{experiment / 'progress.json'}`\n"
        f"Logs: `{experiment / 'logs'}`\n"
    )
    return manifest


def verify_sources(experiment):
    manifest = read(Path(experiment) / "manifest.json")
    for relative, expected in manifest["source_hashes"].items():
        if digest(Path(experiment) / "source_snapshot" / relative) != expected:
            raise ValueError(f"Sealed executable source changed: {relative}")
    for name, expected in manifest["model_files"].items():
        stat = (Path(manifest["model"]) / name).stat()
        if stat.st_size != expected["size"] or stat.st_mtime_ns != expected["mtime_ns"]:
            raise ValueError(f"Frozen model file changed after SHA256 verification: {name}")
    return manifest


def load_graph(experiment, entry, harness):
    from hrp_miniwob.adapters.miniwob_harnesses import Graph, Task
    from hrp_miniwob.harnesses.miniwob_harnesses import Candidate
    from hrp_miniwob.v2.browser import Task as NativeTask

    experiment = Path(experiment)
    path = experiment / entry["task_path"]
    if digest(path) != entry["task_sha256"]:
        raise ValueError("Frozen task changed")
    native = NativeTask(**read(path))
    if native.fingerprint != entry["content_sha256"]:
        raise ValueError("Native task fingerprint changed")
    candidate = Candidate(experiment / "source_snapshot/environment/hrp_miniwob/harnesses" / f"miniwob_{harness.lower()}.py")
    task = Task(**asdict(native), candidate=candidate)
    source = entry["graphs"][harness]
    path = experiment / source["path"]
    if digest(path) != source["sha256"]:
        raise ValueError("Frozen harness graph changed")
    with np.load(path) as saved:
        arrays = {k: saved[k].copy() for k in saved.files}
    graph = Graph(task, HORIZON, candidate, arrays["next_state"], arrays["task_rewards"])
    if (graph.initial_state != int(arrays["initial_state"])
            or tuple(task.actions) != tuple(arrays["semantic_actions"])):
        raise AssertionError("Native harness actions or initial state changed")
    return graph, arrays["bank"], arrays["semantic_codes"]


def render(graph, index):
    chat, codes = graph.candidate.messages(graph.state(int(index)))
    if chat[0]["content"].count(OLD_OBJECTIVE) != 1:
        raise AssertionError("The controlled objective prompt changed unexpectedly")
    chat[0]["content"] = chat[0]["content"].replace(OLD_OBJECTIVE, NEW_OBJECTIVE)
    return chat, codes


def compact_graph(graph, bank):
    remap = np.full(len(graph.next_state), -1, dtype=np.int64)
    remap[bank] = np.arange(len(bank))
    nxt = graph.next_state[bank].copy()
    live = nxt >= 0
    nxt[live] = remap[nxt[live]]
    if np.any(nxt[live] < 0):
        raise AssertionError("The archived state bank is not closed under execution")
    return nxt, graph.task_rewards[bank].copy(), int(remap[graph.initial_state])


def load_policy(manifest):
    import torch

    from hrp_miniwob.adapters.miniwob_harnesses import LanguagePolicy

    torch.set_num_threads(2)
    policy = LanguagePolicy(manifest["model"], seed=42, device="cuda:0",
                            trainable=False, activation_checkpointing=False)
    if any(p.requires_grad for p in policy.model.parameters()):
        raise AssertionError("The 5.1 actor must be frozen")
    policy.model.eval()
    policy.metadata["rank"] = policy.metadata["lora_alpha"] = None
    return policy


def preflight_gpu(experiment):
    import torch

    experiment = Path(experiment)
    manifest = verify_sources(experiment)
    policy = load_policy(manifest)
    checks = []
    with torch.inference_mode():
        for entry in manifest["entries"]:
            if entry["split"] != "calibration":
                continue
            for harness in HARNESSES:
                graph, bank, _ = load_graph(experiment, entry, harness)
                # Cover the initial state and varied controller/budget states.
                indices = sorted({int(bank[0]), int(bank[len(bank)//2]), graph.initial_state})
                prompts, codes = zip(*(render(graph, i) for i in indices), strict=True)
                lp = policy.batch_log_probabilities(prompts, codes).cpu().numpy()
                repeated = policy.batch_log_probabilities(prompts, codes).cpu().numpy()
                error = float(np.max(np.abs(lp - repeated)))
                if error > 1e-10 or not np.isfinite(lp).all():
                    raise AssertionError("Frozen BF16 action probabilities did not replay")
                checks.append({"family": entry["family"], "harness": harness,
                               "states": len(indices), "actions": len(codes[0]),
                               "max_replay_log_error": error})
    result = {"status": "passed", "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "checks": checks, "model_metadata": policy.metadata,
              "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30, "completed_at_utc": now()}
    write(experiment / "acceptance/gpu.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "model_metadata"}), flush=True)


def measure_task(experiment, entry, harness, policy, manifest_hash):
    import torch

    experiment = Path(experiment)
    directory = experiment / "results" / harness
    path = directory / f"{entry['key']}.npz"
    meta_path = path.with_suffix(".json")
    if meta_path.exists():
        old = read(meta_path)
        if old["manifest_sha256"] != manifest_hash or digest(path) != old["table_sha256"]:
            raise ValueError("Resumed result belongs to a different protocol or is corrupt")
        return old
    graph, bank, expected_codes = load_graph(experiment, entry, harness)
    prompts, menus = zip(*(render(graph, index) for index in bank), strict=True)
    if any(tuple(menu) != tuple(expected_codes) for menu in menus):
        raise AssertionError("Action code-to-semantics mapping changed")
    started = time.monotonic()
    before = (policy.forward_observations, policy.forward_prompt_tokens, policy.forward_microbatches)
    with torch.inference_mode():
        lp = policy.batch_log_probabilities(prompts, menus).cpu().numpy()
    nxt, rewards, initial = compact_graph(graph, bank)
    metrics, solved = self_reference_metrics(nxt, rewards, lp, initial, TAU)
    seed = int.from_bytes(hashlib.sha256(f"51942:{entry['key']}:{harness}".encode()).digest()[:8], "little")
    successes, lengths = independent_success_rollouts(nxt, rewards, lp, initial,
                                                     HORIZON, ROLLOUTS, seed)
    if metrics["expected_actor_decisions"] > HORIZON + 1e-10:
        raise AssertionError("Visitation exceeds the decision budget")
    hashes = np.asarray([hashlib.sha256(json.dumps(p, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
                         for p in prompts])
    directory.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, bank=bank, next_state=nxt, task_rewards=rewards,
                            initial_state=initial, log_policy=lp, log_reference=lp.copy(),
                            semantic_actions=np.asarray(graph.task.actions), semantic_codes=expected_codes,
                            prompt_sha256=hashes, rollout_success=successes, rollout_lengths=lengths, **solved)
    tmp.replace(path)
    after = (policy.forward_observations, policy.forward_prompt_tokens, policy.forward_microbatches)
    row = {"task": entry["key"], "family": entry["family"], "seed": entry["seed"],
           "harness": harness, "content_sha256": entry["content_sha256"],
           "manifest_sha256": manifest_hash, "table_sha256": digest(path),
           "source_graph_sha256": entry["graphs"][harness]["sha256"],
           "states": len(bank), "actions": len(expected_codes), **metrics,
           "success_rollout": float(successes.mean()), "rollout_seed": seed,
           "rollout_episodes": ROLLOUTS, "forward_state_rows": after[0]-before[0],
           "prompt_tokens": after[1]-before[1], "forward_microbatches": after[2]-before[2],
           "wall_seconds": time.monotonic()-started, "completed_at_utc": now()}
    write(meta_path, row)
    return row


def worker(experiment, shard):
    experiment = Path(experiment)
    manifest = verify_sources(experiment)
    physical_gpu = GPUS[shard]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != manifest["gpu_inventory"][str(physical_gpu)]["uuid"]:
        raise ValueError("Worker is not isolated on its assigned physical GPU")
    tasks = [e for e in manifest["entries"] if e["split"] == "test"][shard::len(GPUS)]
    status_path = experiment / "workers" / f"gpu{physical_gpu}.json"
    status = {"status": "loading_model", "physical_gpu": physical_gpu, "pid": os.getpid(),
              "completed_evaluations": 0, "total_evaluations": len(tasks)*len(HARNESSES),
              "started_at_utc": now()}
    write(status_path, status)
    try:
        policy = load_policy(manifest)
        manifest_hash = digest(experiment / "manifest.json")
        write(experiment / "workers" / f"gpu{physical_gpu}_model.json", policy.metadata)
        for harness in HARNESSES:
            for entry in tasks:
                row = measure_task(experiment, entry, harness, policy, manifest_hash)
                status.update(status="running", harness=harness, last_task=entry["key"],
                              completed_evaluations=status["completed_evaluations"]+1,
                              updated_at_utc=now())
                write(status_path, status)
                print(json.dumps({"gpu": physical_gpu, "harness": harness, "task": entry["key"],
                                  "completed": status["completed_evaluations"],
                                  "states": row["states"], "wall_seconds": row["wall_seconds"]}), flush=True)
        status.update(status="complete", completed_at_utc=now())
        write(status_path, status)
    except BaseException as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}", updated_at_utc=now())
        write(status_path, status)
        raise


def ranking_accuracy(scores, success):
    correct, total = 0., 0
    for i in range(len(scores)):
        for j in range(i):
            actual = success[i] - success[j]
            if abs(actual) <= 1e-8:
                continue
            predicted = scores[i] - scores[j]
            correct += .5 if abs(predicted) <= 1e-8 else float(actual * predicted > 0)
            total += 1
    return {"accuracy": correct/total if total else None, "non_tied_pairs": total}


def analyze(experiment):
    experiment = Path(experiment)
    manifest = verify_sources(experiment)
    manifest_hash = digest(experiment / "manifest.json")
    tasks = [e for e in manifest["entries"] if e["split"] == "test"]
    names = ("ceiling", "mismatch", "C_minus_M", "J", "success_exact", "success_rollout")
    values = np.empty((len(tasks), len(HARNESSES), len(names)))
    rows = []
    for i, entry in enumerate(tasks):
        for j, harness in enumerate(HARNESSES):
            path = experiment / "results" / harness / f"{entry['key']}.npz"
            row = read(path.with_suffix(".json"))
            if row["manifest_sha256"] != manifest_hash or digest(path) != row["table_sha256"]:
                raise ValueError("Result integrity check failed")
            with np.load(path) as saved:
                if not np.array_equal(saved["log_policy"], saved["log_reference"]):
                    raise AssertionError("Self-reference probabilities differ")
                measured, _ = self_reference_metrics(saved["next_state"], saved["task_rewards"],
                                                    saved["log_policy"], int(saved["initial_state"]), TAU)
                for key in ("ceiling", "mismatch", "J", "success_exact"):
                    if abs(measured[key] - row[key]) > 1e-10:
                        raise AssertionError("Stored metric does not match its policy table")
                closed_form = TAU * np.log1p(measured["success_exact"] * np.expm1(1/TAU))
                if abs(closed_form - measured["ceiling"]) > 1e-10:
                    raise AssertionError("Deterministic task ceiling disagrees with its independent closed form")
                if abs(float(saved["rollout_success"].mean()) - row["success_rollout"]) > 1e-12:
                    raise AssertionError("Stored rollout summary changed")
            values[i, j] = [row[k] for k in names]
            rows.append(row)
    rng = np.random.default_rng(manifest["bootstrap_seed"])
    points, choices, family_bootstraps = [], [], []
    for family in (*FAMILIES, "all"):
        indices = [i for i, e in enumerate(tasks) if e["family"] == family] if family != "all" else list(range(len(tasks)))
        subset = values[indices]
        mean = subset.mean(axis=0)
        if family != "all":
            draws = rng.integers(0, len(indices), size=(manifest["bootstrap_repeats"], len(indices)))
            bootstrap = subset[draws].mean(axis=1)
            family_bootstraps.append(bootstrap)
        else:
            bootstrap = np.mean(family_bootstraps, axis=0)
        bounds = np.quantile(bootstrap, [.025, .975], axis=0)
        for j, harness in enumerate(HARNESSES):
            points.append({"family": family, "harness": harness, "n_tasks": len(indices),
                           **dict(zip(names, mean[j].tolist(), strict=True)),
                           "ci": {k: bounds[:, j, n].tolist() for n, k in enumerate(names)}})
        truth = mean[:, names.index("success_exact")]
        actual = mean[:, names.index("success_rollout")]
        score_sets = {"C": mean[:, 0], "-M": -mean[:, 1], "C-M": mean[:, 2]}
        for label, scores in score_sets.items():
            selected = int(np.argmax(scores))
            choices.append({"family": family, "score": label, "selected_harness": HARNESSES[selected],
                            "exact_best_harness": HARNESSES[int(np.argmax(truth))],
                            "selected_success_exact": float(truth[selected]),
                            "selected_success_rollout": float(actual[selected]),
                            "selection_regret": float(truth.max()-truth[selected]),
                            "ordering": ranking_accuracy(scores, truth),
                            "ranked_harnesses": [HARNESSES[k] for k in np.argsort(-scores, kind="stable")]})
    result = {"status": "complete", "manifest_sha256": manifest_hash, "model": manifest["model"],
              "objective": manifest["objective"], "harnesses": list(HARNESSES), "points": points,
              "selection": choices, "task_evaluations": len(rows),
              "max_gap_identity_error": max(r["gap_identity_error"] for r in rows),
              "max_success_identity_error": max(r["success_identity_error"] for r in rows),
              "total_forward_states": sum(r["forward_state_rows"] for r in rows),
              "total_prompt_tokens": sum(r["prompt_tokens"] for r in rows),
              "rollout_episodes": len(rows)*ROLLOUTS, "completed_at_utc": now()}
    write(experiment / "analysis/section51_data.json", result)
    write(experiment / "analysis/task_results.json", rows)
    lines = ["# Section 5.1: Qwen3.5-9B self-reference", "", "Status: complete and validated.", "",
             ("576 tasks, 192 per family, H0–H5. Frozen off-the-shelf 9B with no project adapter. "
              "Finite horizon 10, no discount or explicit cost, tau=0.1, actor equals reference."), "",
             ("H4 uses the archived frozen 2B reviewer. Prompts use task-success objective wording. "
             "Exact success and the independent 256 graph-rollout success rate are reported separately. "
             "These are native-audited graph evaluations, not new browser trajectories. "
              "C-M=success follows from the configured objective. "
              "At the individual-task level C=tau*log(1+S*(exp(1/tau)-1)), so C is already monotone "
              "in success. Aggregated ceiling averages this nonlinear function across tasks."), "",
             "| Harness | C | M | J=C-M | Exact success | Sampled success |",
             "|---|---:|---:|---:|---:|---:|"]
    for p in points:
        if p["family"] == "all":
            lines.append(f"| {p['harness']} | {p['ceiling']:.6f} | {p['mismatch']:.6f} | "
                         f"{p['J']:.6f} | {p['success_exact']:.2%} | {p['success_rollout']:.2%} |")
    lines += ["", "| Selection rule | Selected harness | Exact success | Regret |",
              "|---|---|---:|---:|"]
    for item in choices:
        if item["family"] == "all":
            lines.append(f"| {item['score']} | {item['selected_harness']} | "
                         f"{item['selected_success_exact']:.2%} | {item['selection_regret']:.2%} |")
    lines += ["", ("Intervals in machine-readable results use 2,000 paired task bootstrap draws, "
              "with equal family weights. No training-seed variation is estimated. "
              "Selection is an oracle comparison on the fixed evaluation distribution. "
              "Finite-estimator performance belongs to Section 5.2 and was not run here."), "",
              f"Maximum identity residual: {result['max_gap_identity_error']:.3g}.", "",
              f"Manifest: `{experiment / 'manifest.json'}`",
              f"Data: `{experiment / 'analysis/section51_data.json'}`"]
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("points", "selection")}), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=EXPERIMENT)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "preflight-gpu", "analyze"):
        commands.add_parser(name)
    run = commands.add_parser("worker")
    run.add_argument("--shard", type=int, choices=range(4), required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.experiment)
        print(json.dumps({"status": "prepared", "test_tasks": result["test_tasks"],
                          "harnesses": result["harnesses"], "physical_gpus": result["physical_gpus"]}))
    elif args.command == "preflight-gpu":
        preflight_gpu(args.experiment)
    elif args.command == "worker":
        worker(args.experiment, args.shard)
    else:
        analyze(args.experiment)


if __name__ == "__main__":
    main()
