"""Shared model, cache and evaluation helpers for the paper experiments."""

import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from hrp_miniwob.browser import CheckboxBrowser
from hrp_miniwob.lm_policy import LanguagePolicy
from hrp_miniwob.suite import load_suite

from hrp_miniwob.adapters.meta_miniwob import (
    Candidate,
    MetaAdapter,
    MetaBrowser,
    evaluate_models,
)
from hrp_miniwob.adapters.native_miniwob import write_json

ROOT = Path(__file__).resolve().parents[3]
BASELINE = ROOT / "environment/hrp_miniwob/harnesses/miniwob_meta_seed.py"
DEFAULT_EXPERIMENT = ROOT / "environment/artifacts/miniwob_meta_qwen35_08b"
SPLITS = ("search", "train", "calibration", "test", "ood")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def model_files(path):
    files = sorted(p for p in Path(path).iterdir() if p.suffix in (".json", ".safetensors", ".jinja", ".txt"))
    if not any(p.suffix == ".safetensors" for p in files):
        raise FileNotFoundError(f"No downloaded model weights in {path}")
    return {p.name: digest(p) for p in files}


def source_hashes():
    files = [ROOT / "environment/hrp_miniwob/adapters/meta_miniwob.py", Path(__file__),
             ROOT / "environment/hrp_miniwob/adapters/verl_miniwob.py", ROOT / "environment/hrp_miniwob/adapters/native_miniwob.py",
             ROOT / "environment/hrp_miniwob/harnesses/miniwob_controlled.py", ROOT / "environment/training/uv.lock"]
    files += list((ROOT / "environment/hrp_miniwob").glob("*.py"))
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(files)}


def read_config(experiment):
    experiment = Path(experiment).resolve()
    config = json.loads((experiment / "config.json").read_text())
    for relative, key in (("suite/manifest.json", "suite_manifest_sha256"),
                          ("splits.json", "splits_sha256"), ("train_public.json", "train_public_sha256")):
        if digest(experiment / relative) != config[key]:
            raise ValueError(f"Prepared data changed: {relative}")
    if (config["gamma"], config["tau"]) != (.95, .1):
        raise ValueError("The shared semantic training engine requires gamma=.95, tau=.1")
    for path, expected in config["model_sha256"].items():
        if model_files(path) != expected:
            raise ValueError(f"Prepared model checkpoint changed: {path}")
    return config


def models_for(experiment, split, limit=None):
    partitions = json.loads((experiment / "splits.json").read_text())
    selected = partitions[split][:limit]
    chosen = set(selected)
    base_split = "train" if split == "search" else split
    models = {m.task.seed: m for m in load_suite(experiment / "suite", base_split) if m.task.seed in chosen}
    if len(models) != len(selected) or not models:
        raise ValueError("Split is empty or disagrees with the browser-verified suite")
    return [models[seed] for seed in selected]


def load_frozen(config, device):
    models = {}
    for role in ("reference_model", "reviewer_model"):
        path = config[role]
        if path not in models:
            models[path] = LanguagePolicy(path, device=device, trainable=False)
            if any(p.requires_grad for p in models[path].model.parameters()):
                raise AssertionError("Auxiliary model must be immutable")
    return models[config["reference_model"]], models[config["reviewer_model"]]


def adapter_for(policy, candidate, frozen, config, experiment):
    # Model hashes and candidate identity namespace every frozen decoder cache.
    identity = {"candidate": candidate.sha256, "models": config["model_sha256"],
                "reference": config["reference_model"], "reviewer": config["reviewer_model"],
                "source": source_hashes()}
    namespace = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return MetaAdapter(policy, candidate, *frozen, experiment / "frozen" / namespace)


def public_traces(models, adapter, output, seed, horizon, limit=8):
    rng, traces = np.random.default_rng(seed), []
    with CheckboxBrowser(horizon=horizon, click_cost=0, submit_cost=0) as browser:
        harness = MetaBrowser(browser, adapter)
        with torch.no_grad():
            for model in models[:limit]:
                graph = adapter.graph(model)
                state, steps = harness.reset(model.task.seed), []
                while not state.terminated:
                    chat, codes = adapter.messages(state)
                    lp = adapter.policy.batch_log_probabilities([chat], [codes])[0].detach().cpu().numpy()
                    action = int(rng.choice(len(codes), p=np.exp(lp)))
                    result = harness.step(state.task.actions[action])
                    index = graph.index(state.remaining, state.memory, state.mask)
                    successor = -1 if result.state.terminated else graph.index(
                        result.state.remaining, result.state.memory, result.state.mask)
                    if successor != graph.next_state[index, action] or result.reward != graph.task_rewards[index, action]:
                        raise AssertionError("Candidate's sampled browser execution disagrees with its compiled graph")
                    steps.append({"observation": json.loads(chat[-1]["content"]),
                                  "semantic_action": state.task.actions[action],
                                  "action_probabilities": dict(zip(codes, np.exp(lp).tolist(), strict=True)),
                                  "reward": result.reward, "reason": result.reason,
                                  "review": harness.last_review})
                    state = result.state
                traces.append({"task_seed": model.task.seed, "steps": steps})
    write_json(output / "traces.json", {"source": "sampled actual browser rollouts; no optimal-policy labels",
                                       "traces": traces, "execution_calls": harness.counts})
    return harness.counts


def evaluate(args):
    experiment, output = args.experiment.resolve(), args.output.resolve()
    config, candidate = read_config(experiment), Candidate(args.candidate)
    output.mkdir(parents=True, exist_ok=True)
    identity = {"config_sha256": digest(experiment / "config.json"), "candidate_sha256": candidate.sha256,
                "adapter_sha256": model_files(args.adapter) if args.adapter else None,
                "split": args.split, "limit": args.limit, "source_sha256": source_hashes(), "device": args.device}
    record = output / "evaluation.json"
    if record.exists() and json.loads(record.read_text()) != identity:
        raise ValueError("Evaluation output belongs to a different checkpoint, candidate or data selection")
    write_json(record, identity)
    if (output / "metrics.json").exists() and (output / "traces.json").exists():
        print((output / "metrics.json").read_text(), flush=True)
        return
    actor = LanguagePolicy(config["actor_model"], seed=config["seed"], device=args.device,
                           adapter_path=str(args.adapter) if args.adapter else None,
                           trainable=False, rank=config["rank"])
    frozen = load_frozen(config, args.device)
    adapter = adapter_for(actor, candidate, frozen, config, experiment)
    started = time.monotonic()
    try:
        models = models_for(experiment, args.split, args.limit)
        calls = public_traces(models, adapter, output, config["seed"], config["horizon"])
        report = evaluate_models(models, adapter, output)
        report["accounting"] = {"wall_seconds": time.monotonic() - started,
                                "rollout_execution_calls": calls,
                                "actor_forwards": actor.forward_observations,
                                "actor_prompt_tokens": actor.forward_prompt_tokens,
                                "reference_requests": adapter.frozen.requests,
                                "reference_forwards": adapter.frozen.forwards,
                                "reviewer_requests": adapter.reviewer.requests,
                                "reviewer_forwards": adapter.reviewer.forwards}
        write_json(output / "metrics.json", report)
        print(json.dumps({"output": str(output), "metrics": report["metrics"]}), flush=True)
    finally:
        adapter.close()


def cross_evaluate(args):
    experiment, output = args.experiment.resolve(), args.output.resolve()
    config = read_config(experiment)
    candidates = [Candidate(path) for path in args.candidates]
    runs = [json.loads((path / "run.json").read_text()) for path in args.train_runs]
    if any(run["status"] != "complete" for run in runs):
        raise ValueError("Cross-evaluation requires completed, fixed-budget training runs")
    for key in ("config_sha256", "method", "seed", "updates", "checkpoint_interval",
                "rollout_episodes", "limit", "source_sha256"):
        if len({json.dumps(run["identity"][key]) for run in runs}) != 1:
            raise ValueError(f"Cross-evaluation arms do not share {key}")
    if runs and runs[0]["identity"]["config_sha256"] != digest(experiment / "config.json"):
        raise ValueError("Training and deployment use different experimental objectives")
    for path, run in zip(args.train_runs, runs, strict=True):
        last = run["checkpoints"][-1]
        checkpoint = path.resolve() / last["directory"]
        if last["updates"] != run["identity"]["updates"] or Path(run["final_adapter"]).resolve() != checkpoint / "actor/adapter":
            raise ValueError("The selected weights are not the declared final-budget checkpoint")
        for name, expected in last["files_sha256"].items():
            if digest(checkpoint / name) != expected:
                raise ValueError(f"Final checkpoint changed before cross-evaluation: {name}")
    output.mkdir(parents=True, exist_ok=True)
    selection = {"candidates": [c.sha256 for c in candidates], "train_runs": [str(p.resolve()) for p in args.train_runs],
                 "train_run_sha256": [digest(p / "run.json") for p in args.train_runs],
                 "split": args.split, "limit": args.limit, "config_sha256": digest(experiment / "config.json"),
                 "source_sha256": source_hashes(), "device": args.device}
    seal = output / "selection.json"
    if seal.exists() and json.loads(seal.read_text()) != selection:
        raise ValueError("Previously sealed cross-evaluation selection would change")
    write_json(seal, selection)
    completion = output / "completion.json"
    if completion.exists():
        if json.loads(completion.read_text())["matrix_sha256"] != digest(output / "matrix.json"):
            raise ValueError("Completed cross-evaluation matrix changed")
        print(json.dumps({"matrix": str(output / "matrix.json"), "status": "already_complete"}), flush=True)
        return
    frozen, rows = load_frozen(config, args.device), []
    models = models_for(experiment, args.split, args.limit)
    checkpoints = [("initial", None)] + [(f"adapted_{i}", run["final_adapter"]) for i, run in enumerate(runs)]
    for label, checkpoint in checkpoints:
        policy = LanguagePolicy(config["actor_model"], device=args.device, trainable=False,
                                adapter_path=checkpoint, rank=config["rank"])
        for i, candidate in enumerate(candidates):
            adapter = adapter_for(policy, candidate, frozen, config, experiment)
            try:
                result = evaluate_models(models, adapter, output / label / f"harness_{i}")
                rows.append({"weights": label, "checkpoint": checkpoint, "harness": str(candidate.path),
                             "candidate_sha256": candidate.sha256, **result["metrics"]})
            finally:
                adapter.close()
        del adapter, policy
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    for candidate in candidates:
        ceilings = [row["ceiling"] for row in rows if row["candidate_sha256"] == candidate.sha256]
        if max(ceilings) - min(ceilings) > 1e-10:
            raise AssertionError("Ceiling changed while only actor weights changed")
    write_json(output / "matrix.json", {"selection": selection, "rows": rows,
                                        "claim": "Fixed-reference C and independent occupancy/KL M; J=C-M"})
    write_json(completion, {"matrix_sha256": digest(output / "matrix.json")})
    print(json.dumps({"matrix": str(output / "matrix.json"), "cells": len(rows)}), flush=True)
