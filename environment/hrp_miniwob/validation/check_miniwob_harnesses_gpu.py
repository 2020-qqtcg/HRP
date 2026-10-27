"""Discarded real-model Exact-PG acceptance for one finite harness."""

import argparse
import hashlib
import json
import os
import socket
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from hrp_miniwob.adapters.miniwob_v2 import microbatches
from hrp_miniwob.harnesses.miniwob_harnesses import PROFILES
from hrp_miniwob.suite import reachable_states
from hrp_miniwob.v2.suite import load_suite
from hrp_miniwob.adapters import miniwob_harnesses as adapter
from hrp_miniwob.adapters.miniwob_v2_batches import exact_batch_update
from hrp_miniwob.adapters.native_miniwob import write_json
from hrp_miniwob.benchmarks import miniwob_harnesses as study


def composition_check(actor, rows):
    """Change companions, row positions and tail lengths at a fixed policy."""
    policy = actor.engine.policy
    prompts, menus = [r[0] for r in rows], [r[1] for r in rows]
    with torch.no_grad(), actor.engine.eval_mode():
        old = [policy.batch_log_probabilities([q], [c])[0].cpu().numpy() for q, c in rows]
    max_log = 0.
    with actor.engine.train_mode():
        for indices, _ in microbatches(policy, prompts, menus):
            for ids in (indices, list(reversed(indices)), indices[:1], indices[:3]):
                lp = policy.batch_log_probabilities([prompts[i] for i in ids], [menus[i] for i in ids])
                max_log = max(max_log, float(np.abs(lp.detach().cpu().numpy() - np.asarray([old[i] for i in ids])).max()))
                del lp
    assert max_log <= 1e-3, max_log
    return {"states": len(rows), "max_log_probability_error": max_log,
            "tested": ["mixed companions", "reversed row order", "one-row tail", "three-row tail", "eval versus train"]}


def checkpoint_gradient_check(actor, bound, models):
    """Compare actual parameter gradients with/without activation recomputation."""
    model = min(models, key=lambda m: (len(m.task.actions), len(json.dumps(m.task.data))))
    graph = bound.graph(model, with_reference=False)
    prompt, codes = bound.messages(graph.state(graph.initial_state))
    gradients, probabilities = [], []
    # Repeated uncheckpointed BF16 backward is itself not bitwise reproducible
    # on this CUDA kernel path. Measure that noise before judging recomputation.
    for enabled in (False, False, True, True):
        if enabled:
            actor.engine.policy.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            actor.engine.policy.model.gradient_checkpointing_disable()
        actor.engine.optimizer_zero_grad()
        lp = actor.engine.policy.batch_log_probabilities([prompt] * 8, [codes] * 8)
        probabilities.append(lp.detach().cpu())
        loss = (lp.exp() * torch.arange(len(codes), device=lp.device)).sum(-1).mean()
        loss.backward()
        gradients.append(torch.cat([p.grad.detach().flatten().cpu() if p.grad is not None
                                    else torch.zeros(p.numel()) for p in actor.engine.parameters]))
        del lp, loss
    actor.engine.optimizer_zero_grad()
    gradients = [g.double() for g in gradients]
    denominator = gradients[0].norm().clamp_min(1e-12)
    probability_error = max(float((probabilities[0] - p).abs().max()) for p in probabilities[1:])
    baseline_error = float((gradients[0] - gradients[1]).norm() / denominator)
    checkpoint_repeat_error = float((gradients[2] - gradients[3]).norm() / denominator)
    gradient_error = max(float((g - baseline).norm() / denominator) for g in gradients[2:] for baseline in gradients[:2])
    tolerance = min(torch.finfo(torch.bfloat16).eps, max(1e-5, 2 * baseline_error))
    assert probability_error <= 1e-3 and gradient_error <= tolerance, (probability_error, gradient_error, baseline_error, tolerance)
    return {"task": model.task.key, "max_log_probability_error": probability_error,
            "relative_parameter_gradient_error": gradient_error, "baseline_repeat_relative_error": baseline_error,
            "checkpoint_repeat_relative_error": checkpoint_repeat_error, "gradient_relative_tolerance": tolerance,
            "criterion": "at most twice measured uncheckpointed BF16 repeat error, capped at one BF16 epsilon; floor 1e-5",
            "passed": True, "activation_checkpointing": "non-reentrant"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", choices=PROFILES, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="models/Qwen3.5-2B")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    os.environ.update(WORLD_SIZE="1", RANK="0", LOCAL_WORLD_SIZE="1", LOCAL_RANK="0",
                      MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    started = time.monotonic()
    report = {"status": "running", "harness": args.harness, "model": args.model,
              "seed": 42, "rank": 16, "batch_size": 4, "microbatch_states": 8,
              "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "source_sha256": study.source_hashes(),
              "scope": "Two discarded GPU updates; numerical replay, full task gradients and long-menu stress. Not a formal result.",
              "updates": []}

    def stage(name):
        report.update(stage=name, wall_seconds=time.monotonic() - started)
        write_json(args.output, report)
        print(json.dumps({"stage": name, "seconds": report["wall_seconds"]}), flush=True)

    bound = None
    try:
        with tempfile.TemporaryDirectory(prefix="hrp-harness-gpu-") as cache, study.bindings(), study.base.bindings():
            from hrp_miniwob.adapters.verl_miniwob import make_worker
            stage("loading")
            actor = make_worker(args.model, "actor", "cuda:0", 42, 1e-5, 16)
            frozen = adapter.LanguagePolicy(args.model, trainable=False, device="cuda:0", rank=16)
            assert actor.engine.policy.metadata["dtype"] == "bfloat16"
            assert all(p.dtype == torch.float32 for p in actor.engine.parameters)
            assert not any(p.requires_grad for p in frozen.model.parameters())
            bound = adapter.Adapter(actor.engine.policy, adapter.Candidate(study.candidate_path(args.harness)),
                                    frozen, frozen, cache)
            training = load_suite(args.suite, "train")
            stage("activation_gradient_equivalence")
            report["activation_checkpointing"] = checkpoint_gradient_check(actor, bound, training)
            families = sorted({m.task.family for m in training})
            chosen = [max((m for m in training if m.task.family == family), key=lambda m: len(m.task.actions))
                      for family in families]
            chosen.append(next(m for m in training if m.task.family == "ascending-numbers"
                               and all(m.task != other.task for other in chosen)))
            rows, graphs = [], []
            for model in chosen:
                graph = bound.graph(model, with_reference=False)
                bank = reachable_states(graph)
                rows.extend(bound.messages(graph.state(int(i))) for i in bank[::max(1, len(bank) // 8)])
                graphs.append({"task": model.task.key, "states": len(bank), "actions": len(graph.task.actions),
                               "controller_states": graph.task.memories})
            stage("initial_replay")
            report["graphs"] = graphs
            report["initial_composition"] = composition_check(actor, rows)
            with torch.no_grad():
                error = max(float(np.abs(actor.engine.policy.batch_log_probabilities([q], [c])[0].cpu().numpy()
                                         - bound.frozen(q, c)).max()) for q, c in rows)
            assert error <= 1e-3
            report["initial_reference_error"] = error
            initial_parameters = hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in actor.engine.parameters)).hexdigest()
            report["initial_trainable_sha256"] = initial_parameters
            torch.cuda.reset_peak_memory_stats()
            for number in range(2):
                stage(f"exactpg_update_{number + 1}")
                tick = time.monotonic()
                result = exact_batch_update(actor, chosen, bound)
                report["updates"].append(result | {"seconds": time.monotonic() - tick})
            assert actor.engine.updates == 2
            assert initial_parameters != hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in actor.engine.parameters)).hexdigest()
            stage("trained_replay")
            report["trained_composition"] = composition_check(actor, rows)
            stage("large_menu_stress")
            stress = []
            for family in families:
                model = max((m for m in load_suite(args.suite, "ood") if m.task.family == family),
                            key=lambda m: (len(m.task.actions), len(json.dumps(m.task.data))))
                graph = bound.graph(model, with_reference=False)
                for physical in (0, model.task.physical_states - 1):
                    state = graph.state(graph.index(10, graph.task.memories - 1, physical))
                    q, c = bound.messages(state)
                    actor.engine.optimizer_zero_grad()
                    values = actor.engine.policy.batch_log_probabilities([q] * 8, [c] * 8)
                    values.mean().backward()
                    del values
                    actor.engine.optimizer_zero_grad()
                    stress.append({"task": model.task.key, "actions": len(c),
                                   "tokens": len(actor.engine.policy.tokenizer.encode(actor.engine.policy.chat_text(q)))})
            report.update(status="passed", stress=stress, actor_updates=actor.engine.updates,
                          dtype="bfloat16", trainable_dtype="float32",
                          max_memory_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                          max_memory_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
    except Exception as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        if bound:
            bound.close()
        report["wall_seconds"] = time.monotonic() - started
        write_json(args.output, report)
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    print(json.dumps({"status": report["status"], "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
