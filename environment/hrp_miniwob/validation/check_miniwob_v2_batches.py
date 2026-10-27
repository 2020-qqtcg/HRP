"""CPU numerical batching and resume checks; no real LM training or GPU use."""

import argparse
import json
import sys
import tempfile
from contextlib import ExitStack, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from hrp_miniwob.validation.check_miniwob_meta_harness import ProbePolicy
from hrp_miniwob.oracle import evaluate
from hrp_miniwob.v2.model import compile_task
from hrp_miniwob.v2 import FAMILIES
from hrp_miniwob.v2.browser import Browser

from hrp_miniwob.adapters import verl_miniwob
from hrp_miniwob.adapters.miniwob_v2 import Adapter, Candidate, exact_update, start_bank
from hrp_miniwob.adapters.miniwob_v2_batches import (
    exact_batch_update,
)
from hrp_miniwob.adapters.native_miniwob import write_json
from hrp_miniwob.benchmarks import miniwob_meta_harness as host
from hrp_miniwob.benchmarks.miniwob_v2_train import resolve_budget, train


class Policy(ProbePolicy):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.register_parameter("logits", torch.nn.Parameter(torch.linspace(-.2, .2, 26, dtype=torch.float64)))
        self.forward_observations = self.gradient_observations = 0
        self.metadata = {}

    def batch_log_probabilities(self, chats, codes):
        self.forward_observations += len(chats)
        self.forward_prompt_tokens += len(json.dumps(chats))
        self.gradient_observations += len(chats) * int(torch.is_grad_enabled())
        return torch.stack([self.model.logits[:len(menu)].log_softmax(-1) for menu in codes])


class Worker:
    """CPU parameters and optimizer for exact-gradient and checkpoint checks."""
    def __init__(self, path=None, role="actor", device="cuda:mock", seed=42, lr=.01, rank=16):
        assert device == "cuda:mock"
        torch.manual_seed(seed)
        self.engine, self.role, self.policy, self.updates = self, role, Policy(), 0
        self.optimizer = torch.optim.SGD(self.policy.model.parameters(), lr=lr, momentum=.9)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda _: 1.)
        self.profiler = SimpleNamespace(step=lambda: None)
        self.seen = []
        self.gradient = None
        self.no_step = False


    def train_mode(self, **kwargs):
        return nullcontext()

    eval_mode = train_mode


    def optimizer_zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    def optimizer_step(self):
        self.gradient = self.policy.model.logits.grad.detach().clone()
        norm = float(self.gradient.norm())
        if not self.no_step:
            self.optimizer.step()
        self.updates += 1
        return norm

    def lr_scheduler_step(self):
        if not self.no_step:
            self.scheduler.step()


    def save_checkpoint(self, directory, global_step):
        path = Path(directory)
        (path / "adapter").mkdir(parents=True, exist_ok=True)
        assert global_step == self.updates
        torch.save({"logits": self.policy.model.logits.detach(), "updates": self.updates,
                    "optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict()}, path / "state.pt")
        torch.save(self.policy.model.state_dict(), path / "adapter/weights.pt")

    def load_checkpoint(self, directory, **kwargs):
        state = torch.load(Path(directory) / "state.pt", weights_only=True)
        with torch.no_grad():
            self.policy.model.logits.copy_(state["logits"])
        self.updates = state["updates"]
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])


def assert_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            assert_equal(x, y)
    else:
        assert a == b


def checks(directory):
    with Browser(horizon=4) as browser:
        models = [compile_task(browser.reset(42, family).task, horizon=4) for family in FAMILIES]
    # Different graph sizes distinguish task averaging from state pooling.
    batch = [models[i] for i in (0, 1, 2, 0)]
    candidate = Candidate(ROOT / "environment/hrp_miniwob/harnesses/miniwob_v2_seed.py")
    worker = Worker()
    worker.no_step = True
    bound = Adapter(worker.policy, candidate, ProbePolicy(.03), ProbePolicy(.1), directory / "gradient_cache")
    try:
        gradients = []
        for model in batch:
            exact_update(worker, model, bound)
            gradients.append(worker.gradient.clone())
        previous = worker.updates
        record = exact_batch_update(worker, batch, bound)
        error = float((worker.gradient - torch.stack(gradients).mean(0)).abs().max())
        assert error < 1e-12 and worker.updates == previous + 1
        analytical = worker.gradient.clone()
        initial = worker.policy.model.logits.detach().clone()
        direction = torch.linspace(-.5, .5, len(initial), dtype=initial.dtype).roll(3)

        def objective(offset):
            with torch.no_grad():
                worker.policy.model.logits.copy_(initial + offset * direction)
            values = []
            for model in batch:
                graph, lp = bound.measure(model)
                mu, _, _ = start_bank(graph)
                values.append(float(mu @ evaluate(graph.next_state, graph.task_rewards, graph.reference(), np.exp(lp), graph.initial_state, .95, .1).value))
            return float(np.mean(values))

        fd = (objective(1e-5) - objective(-1e-5)) / 2e-5
        direction_error = abs(fd + float(analytical @ direction))  # loss = -J
        assert direction_error < 1e-8
        with torch.no_grad():
            worker.policy.model.logits.copy_(initial)
        exact_batch_update(worker, [batch[0]] * 4, bound)
        duplicate_error = float((worker.gradient - gradients[0]).abs().max())
        assert duplicate_error < 1e-12
        exact_batch_update(worker, [batch[0]], bound)
        torch.testing.assert_close(worker.gradient, gradients[0], atol=1e-12, rtol=1e-12)
        exact_batch_update(worker, batch, bound, microbatch_size=1)
        accumulation_error = float((worker.gradient - analytical).abs().max())
        assert accumulation_error < 1e-12
    finally:
        bound.close()
    budget_rows = []
    for method in ("exactpg",):
        args = SimpleNamespace(method=method, batch_size=None,
                               updates=None, checkpoint_interval=None, rollout_episodes=None)
        settings = resolve_budget(args)
        assert settings["forward_microbatch_states"] == 8
        assert args.batch_size == 4
        assert args.updates == 1024 and args.checkpoint_interval == 64
        total = args.updates * args.batch_size
        assert total == 4096
        budget_rows.append({**vars(args), **settings, "sampled_episodes_or_task_draws": total})
        # Explicit sizes are counts, accept non-power-of-two Exact-PG batches,
        # and do not silently rescale updates or evaluation spacing.
        args.batch_size = 3
        resolve_budget(args)
        assert args.updates == 1024 and args.checkpoint_interval == 64
    invalid = [{"batch_size": 0}, {"updates": 0}, {"checkpoint_interval": 0}]
    for change in invalid:
        args = SimpleNamespace(**({"method": "exactpg", "batch_size": 4,
                                   "updates": None, "checkpoint_interval": None, "rollout_episodes": None} | change))
        try:
            resolve_budget(args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid batch configuration accepted: {change}")
    resume = []
    for method in ("exactpg",):
        experiment = directory / method
        experiment.mkdir()
        write_json(experiment / "config.json", {"test_only": True})
        write_json(experiment / "train_public.json", [{}])
        failure = {"directory": None, "raised": False}
        config = {"actor_model": "CPU_TEST", "rank": 16, "horizon": 4, "actor_lr": .01}

        def adapter_for(policy, c, frozen, config, experiment):
            return Adapter(policy, c, ProbePolicy(.03), ProbePolicy(.1), experiment / "cache")

        def evaluate_models(models, bound, output, failure=failure):
            if failure["directory"] is not None and output.parent == failure["directory"] and not failure["raised"]:
                failure["raised"] = True
                raise RuntimeError("Intentional checkpoint-boundary interruption")
            return {"metrics": {"success_probability": float(bound.policy.model.logits[0].detach())}}

        budget, interval = (3, 1)
        args = SimpleNamespace(experiment=experiment, output=experiment / "full", method=method,
                               candidate=candidate.path, device="cuda:mock", seed=42, batch_size=None,
                               updates=budget, checkpoint_interval=interval, rollout_episodes=256, limit=None, resume=False)
        args.checkpoint_updates = [0, 1, 3]
        with ExitStack() as stack:
            for name, value in {"Candidate": Candidate, "read_config": lambda p, config=config: config, "models_for": lambda *args: batch,
                                "load_frozen": lambda *args: None, "adapter_for": adapter_for,
                                "evaluate_models": evaluate_models,
                                "CheckboxBrowser": lambda **kwargs: SimpleNamespace(close=lambda: None)}.items():
                stack.enter_context(patch.object(host, name, value))
            stack.enter_context(patch.object(verl_miniwob, "make_worker", Worker))
            train(args)
            full = args.output
            args.output = experiment / "resumed"
            failure["directory"] = args.output / f"checkpoint_{interval:06d}"
            try:
                train(args)
            except RuntimeError as exc:
                assert "Intentional" in str(exc)
            else:
                raise AssertionError("Interruption was not exercised")
            saved_batch = args.batch_size
            args.resume, args.batch_size = True, saved_batch * 2
            try:
                train(args)
            except ValueError as exc:
                assert "batch" in str(exc)
            else:
                raise AssertionError("Resume accepted changed batch settings")
            args.batch_size = saved_batch
            train(args)
            manifest = json.loads((args.output / "run.json").read_text())
            assert [c['updates'] for c in manifest['checkpoints']] == [0, 1, 3]
            assert manifest["identity"]["batch"]["batch_size"] == saved_batch
            for role in (("actor",)):
                left = torch.load(full / f"checkpoint_{budget:06d}" / role / "state.pt", weights_only=True)
                right = torch.load(args.output / f"checkpoint_{budget:06d}" / role / "state.pt", weights_only=True)
                assert_equal(left, right)
            left = json.loads((full / f"checkpoint_{budget:06d}/controller.json").read_text())
            right = json.loads((args.output / f"checkpoint_{budget:06d}/controller.json").read_text())
            for key in ("task_rng", "action_rng", "episodes", "task_draws"):
                assert_equal(left[key], right[key])
            assert_equal(torch.load(full / f"checkpoint_{budget:06d}/rng.pt", weights_only=True),
                         torch.load(args.output / f"checkpoint_{budget:06d}/rng.pt", weights_only=True))
            resume.append({"method": method, "updates": budget, "episodes": right["episodes"], "task_draws": right["task_draws"], "resumed_state_matches": True})
    return {"gradient_mean_error": error, "gradient_direction_error": direction_error,
            "duplicate_task_error": duplicate_error, "microbatch_accumulation_error": accumulation_error,
            "exact_batch": record, "budgets": budget_rows, "resume": resume}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "environment/artifacts/v2/batch_acceptance.json")
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = {"status": "running", "scope": "CPU Exact-PG gradients, task batching, and checkpoint resume"}
    try:
        with tempfile.TemporaryDirectory(prefix="hrp-v2-batch-check-") as directory:
            report.update(checks(Path(directory)))
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        write_json(args.output, report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
