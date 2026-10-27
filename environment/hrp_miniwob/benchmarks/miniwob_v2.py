"""Versioned three-family coverage study and Meta-Harness evaluator entrypoint."""

import argparse
import fcntl
import json
import math
import os
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from hrp_miniwob.v2.browser import Browser
from hrp_miniwob.v2.suite import SPLITS, load_suite, prepare_suite

from hrp_miniwob.adapters import miniwob_v2 as adapter
from hrp_miniwob.adapters.native_miniwob import write_json
from hrp_miniwob.benchmarks import miniwob_meta_harness as runner
from hrp_miniwob.benchmarks.miniwob_v2_train import resolve_budget, train

ROOT = Path(__file__).resolve().parents[3]
BASELINE = ROOT / "environment/hrp_miniwob/harnesses/miniwob_v2_seed.py"
DEFAULT_EXPERIMENT = ROOT / "environment/artifacts/v2/three_tasks"
V1_SOURCE_HASHES = runner.source_hashes


def source_hashes():
    result = V1_SOURCE_HASHES()
    paths = list((ROOT / "environment/hrp_miniwob/v2").glob("*.py")) + [
        ROOT / "environment/hrp_miniwob/v2/upstream.json",
        ROOT / "environment/hrp_miniwob/adapters/miniwob_v2.py", Path(__file__),
        ROOT / "environment/hrp_miniwob/adapters/miniwob_v2_batches.py", ROOT / "environment/hrp_miniwob/benchmarks/miniwob_v2_train.py",
        ROOT / "environment/hrp_miniwob/benchmarks/miniwob_v2.py",
         ROOT / "src/hrp/initial_distribution.py"]
    result.update({str(p.relative_to(ROOT)): runner.digest(p) for p in paths})
    return result


def require_v2_path(path):
    path = Path(path).resolve()
    if "v2" not in path.relative_to(ROOT / "environment/artifacts").parts:
        raise ValueError("v2 data/results must live beneath environment/artifacts/v2 (or an explicit v2 subdirectory)")
    return path


def prepare(args):
    experiment = require_v2_path(args.experiment)
    if (experiment / "config.json").exists():
        raise FileExistsError("Prepared v2 experiments are immutable; use a new directory")
    manifest = prepare_suite(experiment / "suite", args.horizon, args.start_seed, args.profile)
    partitions = {s: [e["seed"] for e in manifest["entries"] if e["split"] == s] for s in SPLITS}
    public = [{"seed": m.task.seed, "task": asdict(m.task), "content_sha256": m.task.fingerprint}
              for m in load_suite(experiment / "suite", "train")]
    write_json(experiment / "splits.json", partitions)
    write_json(experiment / "train_public.json", public)
    probe = defaultdict(list)
    for row in public:
        if len(probe[row["task"]["family"]]) < 24:
            probe[row["task"]["family"]].append(row["seed"])
    write_json(experiment / "train_probe.json", dict(probe))
    roles = {role: str(getattr(args, role).resolve()) for role in ("actor_model", "reference_model", "reviewer_model")}
    config = {"schema_version": 2, "task_version": "v2", "profile": args.profile,
              **roles, "model_sha256": {p: runner.model_files(p) for p in set(roles.values())},
              "seed": args.seed, "rank": 16, "horizon": args.horizon,
              "gamma": .95, "tau": .1, "click_cost": 0., "submit_cost": 0.,
              "actor_lr": 1e-5,
              "suite_manifest_sha256": runner.digest(experiment / "suite/manifest.json"),
              "splits_sha256": runner.digest(experiment / "splits.json"),
              "train_public_sha256": runner.digest(experiment / "train_public.json"),
              "train_probe_sha256": runner.digest(experiment / "train_probe.json"),
              "split_counts": {s: len(v) for s, v in partitions.items()},
              "initial_distribution": "0.5 native + 0.5 legal-prefix coverage; category x budget bin x controller memory, uniform nonempty cells then states",
              "task_weighting": "uniform family then uniform unique instance within family",
              "reference_contract": "Independent frozen checkpoint reads the same new harness prompt as the actor",
              "selection": "Record full curves; E1 uses best native calibration family-macro success, earliest tie; cross-eval uses final fixed-budget checkpoint",
              "families": manifest["families"],
              "claim_scope": "Three controlled native families and explicit public history; this task combination requires its own LM geometry measurements",
              "source_sha256_at_preparation": source_hashes()}
    write_json(experiment / "config.json", config)
    print(json.dumps({"config": str(experiment / "config.json"), "splits": config["split_counts"]}), flush=True)


V1_READ_CONFIG = runner.read_config


def read_config(experiment):
    config = V1_READ_CONFIG(require_v2_path(experiment))
    if config.get("task_version") != "v2":
        raise ValueError("The v2 entrypoint cannot consume a v1 experiment")
    if runner.digest(Path(experiment) / "train_probe.json") != config["train_probe_sha256"]:
        raise ValueError("The fixed v2 training probe changed")
    if config["source_sha256_at_preparation"] != source_hashes():
        raise ValueError("v2 source changed since preparation; prepare a fresh sealed experiment")
    return config


class BalancedTasks:
    """Virtual uniform-family sampling through the unchanged integer sampler."""
    def __init__(self, models):
        groups = defaultdict(list)
        for model in models:
            groups[model.task.family].append(model)
        self.groups = [groups[key] for key in sorted(groups)]
        self.period = math.lcm(*(len(group) for group in self.groups))

    def __len__(self):
        return self.period * len(self.groups)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        family, offset = divmod(index, self.period)
        group = self.groups[family]
        return group[offset % len(group)]


def models_for(experiment, split, limit=None):
    models = load_suite(experiment / "suite", split)
    if not models:
        raise ValueError("v2 split is empty")
    selected = json.loads((experiment / "splits.json").read_text())[split]
    if [m.task.seed for m in models] != selected:
        raise ValueError("v2 split index disagrees with suite manifest")
    if limit:
        groups = defaultdict(list)
        for model in models:
            if len(groups[model.task.family]) < limit:
                groups[model.task.family].append(model)
        models = [m for group in groups.values() for m in group]
    return models


def public_traces(models, bound, output, seed, horizon, limit=8):
    rng, traces = np.random.default_rng(seed), []
    # Include each family before taking additional task instances.
    groups = defaultdict(list)
    for model in models:
        groups[model.task.family].append(model)
    ordered = [group[i] for i in range(max(map(len, groups.values()))) for group in groups.values() if i < len(group)]
    with Browser(horizon=horizon) as browser:
        harness = adapter.Harness(browser, bound)
        with torch.no_grad():
            for model in ordered[:limit]:
                graph = bound.graph(model, with_reference=False)
                state, steps = harness.reset(model.task), []
                while not state.terminated:
                    chat, codes = bound.messages(state)
                    lp = bound.policy.batch_log_probabilities([chat], [codes])[0].detach().cpu().numpy()
                    action = int(rng.choice(len(codes), p=np.exp(lp)))
                    result = harness.step(state.task.actions[action])
                    index = graph.index(state.remaining, state.memory, state.physical)
                    nxt = -1 if result.state.terminated else graph.index(result.state.remaining, result.state.memory, result.state.physical)
                    if nxt != graph.next_state[index, action] or result.reward != graph.task_rewards[index, action]:
                        raise AssertionError("Live candidate execution differs from its v2 finite graph")
                    steps.append({"observation": state.public(), "action": state.task.actions[action],
                                  "probabilities": np.exp(lp).tolist(), "reward": result.reward,
                                  "raw_reward": result.raw_reward, "review": harness.last_review})
                    state = result.state
                traces.append({"family": model.task.family, "seed": model.task.seed, "steps": steps})
        write_json(output / "traces.json", {"traces": traces, "source": "live native browser; public observations only"})
        return harness.counts


@contextmanager
def bindings(training=False):
    """Process-local dependency binding; original v1 files and imports survive."""
    # Eight-state BF16 activations vary by length bucket. Avoid allocator
    # fragmentation before any worker/model allocates CUDA tensors.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from hrp_miniwob.adapters import verl_miniwob
    def sampling_models(experiment, split, limit=None):
        models = models_for(experiment, split, limit)
        return BalancedTasks(models) if training and split == "train" else models

    def training_adapter(policy, candidate, frozen, config, experiment):
        bound = original["adapter_for"](policy, candidate, frozen, config, experiment)
        if training:
            selected = {seed for seeds in json.loads((experiment / "train_probe.json").read_text()).values() for seed in seeds}
            bound.train_probe = [m for m in load_suite(experiment / "suite", "train") if m.task.seed in selected]
        return bound

    def evaluation(models, bound, output):
        report = adapter.evaluate_models(models, bound, output)
        if training and Path(output).name == "calibration":
            adapter.evaluate_models(bound.train_probe, bound, Path(output).parent / "train_probe")
        return report

    replacements = {"Candidate": adapter.Candidate, "MetaAdapter": adapter.Adapter,
                    "LanguagePolicy": adapter.LanguagePolicy, "CheckboxBrowser": Browser,
                    "models_for": sampling_models, "read_config": read_config, "source_hashes": source_hashes,
                    "adapter_for": training_adapter,
                    "public_traces": public_traces, "evaluate_models": evaluation}
    original = {key: getattr(runner, key) for key in replacements}
    policy = verl_miniwob.LanguagePolicy
    worker_factory = verl_miniwob.make_worker

    def make_worker(*args, **kwargs):
        worker = worker_factory(*args, **kwargs)
        worker.engine.engine_config.micro_batch_size_per_gpu = adapter.MICROBATCH_STATES
        worker.engine.engine_config.infer_micro_batch_size_per_gpu = adapter.MICROBATCH_STATES
        worker.engine.policy.metadata["optimizer"] = "verl TrainingWorker; episode/task batch specified by v2 run settings"
        return worker

    for key, value in replacements.items():
        setattr(runner, key, value)
    verl_miniwob.LanguagePolicy = adapter.LanguagePolicy
    verl_miniwob.make_worker = make_worker
    try:
        yield
    finally:
        for key, value in original.items():
            setattr(runner, key, value)
        verl_miniwob.LanguagePolicy = policy
        verl_miniwob.make_worker = worker_factory


def select_checkpoint(output):
    run = json.loads((output / "run.json").read_text())
    candidates = [row for row in run["checkpoints"] if "calibration" in row]
    best = max(candidates, key=lambda row: (row["calibration"]["success_probability"], -row["updates"]))
    write_json(output / "e1_selection.json", {"selection": "highest native calibration family-macro success, earliest tie, includes initialization",
               "updates": best["updates"], "adapter": str(output / best["directory"] / "actor/adapter"),
               "calibration": best["calibration"], "run_sha256": runner.digest(output / "run.json")})


def cross_evaluate(args):
    batches = [json.loads((path / "run.json").read_text())["identity"]["batch"] for path in args.train_runs]
    if len({json.dumps(value, sort_keys=True) for value in batches}) != 1:
        raise ValueError("Cross-evaluation training arms must share the same effective batch settings")
    return runner.cross_evaluate(args)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    p.add_argument("--profile", choices=("full", "smoke"), default="full")
    p.add_argument("--actor-model", type=Path, default=Path("models/Qwen3.5-2B"))
    for role in ("reference", "reviewer"):
        p.add_argument(f"--{role}-model", type=Path, default=Path("models/Qwen3.5-2B"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--start-seed", type=int, default=9100000)
    p.add_argument("--horizon", type=int, default=10)
    for command in ("evaluate", "train", "cross-eval"):
        p = sub.add_parser(command)
        p.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--device", default="cuda:0")
        if command != "train":
            p.add_argument("--limit", type=int, help="Diagnostic instance cap per family")
        if command != "cross-eval":
            p.add_argument("--candidate", type=Path, default=BASELINE)
        if command == "evaluate":
            p.add_argument("--split", choices=SPLITS, default="search")
            p.add_argument("--adapter", type=Path)
        elif command == "train":
            p.add_argument("--method", choices=("exactpg",), default="exactpg")
            p.add_argument("--seed", type=int, default=42)
            p.add_argument("--batch-size", type=int, help="Tasks per Exact-PG update (default 4)")
            p.add_argument("--updates", type=int, help="Optimizer updates, independent of batch size (default 1024)")
            p.add_argument("--checkpoint-interval", type=int, help="Optimizer updates between evaluations (default 64)")
            p.add_argument("--resume", action="store_true")
            p.add_argument("--initial-adapter", type=Path, help="Initialize actor weights from this adapter, with fresh optimizer state")
            p.add_argument("--checkpoint-updates", type=int, nargs="+", help="Explicit increasing checkpoint nodes including zero and the final update")
            p.set_defaults(limit=None)
        else:
            p.add_argument("--candidates", type=Path, nargs="+", required=True)
            p.add_argument("--train-runs", type=Path, nargs="+", required=True)
            p.add_argument("--split", choices=("calibration", "test", "ood"), default="test")
    args = parser.parse_args(argv)
    if getattr(args, "limit", None) is not None and args.limit < 1:
        parser.error("--limit must be positive")
    torch.set_num_threads(2)
    if args.command == "prepare":
        prepare(args)
        return
    if args.command == "train":
        resolve_budget(args)
    args.output = require_v2_path(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".run.lock").open("a") as lock, bindings(training=args.command == "train"):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        {"train": train, "evaluate": runner.evaluate, "cross-eval": cross_evaluate}[args.command](args)
        if args.command == "train":
            select_checkpoint(args.output)


if __name__ == "__main__":
    main()
