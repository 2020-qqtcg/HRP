"""Check real browser transitions, exact gradients, and optional downloaded-model training/reload.

All outputs are implementation smoke artifacts, excluded from experiment data.
"""

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from hrp_miniwob.browser import CheckboxBrowser
from hrp_miniwob.lm_policy import LanguagePolicy
from hrp_miniwob.measurements import policy_gradient_weights, policy_metrics
from hrp_miniwob.model import compile_task
from hrp_miniwob.suite import reachable_states
from safetensors import safe_open

from hrp_miniwob.adapters.meta_miniwob import Candidate, MetaAdapter, MetaBrowser, MetaState
from hrp_miniwob.adapters.native_miniwob import write_json


class ProbePolicy:
    """Nonuniform deterministic test decoder, never used for experimental scores."""

    def __init__(self, tilt=1.):
        self.model = torch.nn.Identity()
        self.forward_prompt_tokens = 0
        self.tilt = tilt

    def batch_log_probabilities(self, chats, codes):
        self.forward_prompt_tokens += sum(len(json.dumps(chat)) for chat in chats)
        return torch.stack([(torch.arange(len(menu), dtype=torch.float64) * self.tilt).log_softmax(-1)
                            for menu in codes])

    def chat_text(self, chat):
        return json.dumps(chat)

    def tokenizer(self, text, **kwargs):
        return {"input_ids": text.split()}


def gradient_check(graph):
    rng = np.random.default_rng(42)
    logits = torch.tensor(rng.normal(size=graph.next_state.shape), requires_grad=True)
    policy = logits.softmax(-1)
    reference = torch.tensor(graph.reference())
    reward = torch.tensor(graph.task_rewards)
    values = []
    for state, successors in enumerate(graph.next_state):
        future = torch.stack([values[int(n)] if n >= 0 else logits.new_tensor(0.) for n in successors])
        values.append((policy[state] * (reward[state] - .1 * (policy[state].log() - reference[state].log()) + .95 * future)).sum())
    values[graph.initial_state].backward()
    p = policy.detach().numpy()
    weights, advantage = policy_gradient_weights(graph, p, np.zeros(p.shape[1]), graph.reference())
    error = float(np.max(np.abs(logits.grad.numpy() - weights[:, None] * p * advantage)))
    if error > 1e-10:
        raise AssertionError(f"Independent differentiated-return check failed: {error}")
    return error


def parameter_hash(policy):
    digest = hashlib.sha256()
    for name, value in policy.model.named_parameters():
        if value.requires_grad:
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def browser_check(output):
    results = []
    # Exercise a generated controller with a state space beyond the old one-slot adapter.
    generated = output / "three_state_candidate.py"
    generated.write_text('''MEMORY_STATES = 3
INITIAL_MEMORY = 2
def render(observation):
    return "Use the public checkbox state and remaining budget."
def review_mode(memory):
    return ["none", "program", "lm_tool"][memory]
def advance(memory, action, approved):
    return max(0, memory - 1) if action == "submit" else memory
''')
    with CheckboxBrowser(horizon=4, click_cost=0, submit_cost=0) as browser:
        seed = 42
        while True:
            task = browser.reset(seed).task
            if len(task.labels) == 2 and task.target_mask.bit_count() == 1:
                break
            seed += 1
        model = compile_task(browser, seed)
        for path in [ROOT / "environment/hrp_miniwob/harnesses/miniwob_meta_seed.py", ROOT / "environment/hrp_miniwob/harnesses/miniwob_meta_verify.py", generated]:
            candidate = Candidate(path)
            adapter = MetaAdapter(ProbePolicy(.7), candidate, ProbePolicy(.2), ProbePolicy(.4), output / path.stem)
            try:
                graph, lp = adapter.measure(model)
                harness, checks = MetaBrowser(browser, adapter), 0
                for index in reachable_states(graph):
                    state = graph.state(int(index))
                    for action_index, action in enumerate(model.task.actions):
                        harness.restore_for_audit(state)
                        before = harness.counts.copy()
                        result = harness.step(action)
                        successor = -1 if result.state.terminated else graph.index(
                            result.state.remaining, result.state.memory, result.state.mask)
                        if successor != graph.next_state[index, action_index] or result.reward != graph.task_rewards[index, action_index]:
                            raise AssertionError("Compiled controller differs from its actual browser execution")
                        if result.state.remaining != state.remaining - 1 or harness.counts["actor_decisions"] - before["actor_decisions"] != 1:
                            raise AssertionError("Harness obtained a free actor step")
                        expected_mask = state.mask if action == "submit" else state.mask ^ (1 << action_index)
                        if result.state.mask != expected_mask:
                            raise AssertionError("Action mapping changed actual checkbox execution")
                        checks += 1
                row = policy_metrics(graph, np.exp(lp), np.zeros(lp.shape[1]), graph.reference())
                if row["gap_identity_error"] > 1e-10:
                    raise AssertionError("Independent C-J and occupancy/KL mismatch disagree")
                memories = sorted({graph.state(int(index)).memory for index in reachable_states(graph)})
                if path == generated and memories != [0, 1, 2]:
                    raise AssertionError("The generated three-state controller was not fully exercised")
                alternative = MetaAdapter(ProbePolicy(-.3), candidate, ProbePolicy(.2), ProbePolicy(.4), output / path.stem)
                try:
                    other, _ = alternative.measure(model)
                    if not np.array_equal(graph.reference(), other.reference()) or not np.array_equal(graph.next_state, other.next_state):
                        raise AssertionError("Changing only the actor altered the reference or environment")
                finally:
                    alternative.close()
                results.append({"candidate": path.name, "memory_states": candidate.memories,
                                "reachable_memory_states": memories,
                                "browser_edges": checks, "gap_identity_error": row["gap_identity_error"],
                                "gradient_max_error": gradient_check(graph)})
            finally:
                adapter.close()
    return results


def model_check(args, output):
    actor = LanguagePolicy(str(args.model), seed=42, device=args.device, rank=16)
    frozen = LanguagePolicy(str(args.reference_model), device=args.device, trainable=False)
    if any(p.requires_grad for p in frozen.model.parameters()):
        raise AssertionError("Reference weights are not frozen")
    with CheckboxBrowser(horizon=3, click_cost=0, submit_cost=0) as browser:
        state = browser.reset(42)
    candidate = Candidate(ROOT / "environment/hrp_miniwob/harnesses/miniwob_meta_seed.py")
    chat, codes = candidate.messages(MetaState(state.task, state.mask, state.remaining, 0))
    original_weights = next(args.model.glob("*.safetensors"))
    base = dict(actor.model.get_base_model().named_parameters())
    checked = []
    with safe_open(original_weights, framework="pt", device="cpu") as tensors:
        for name in ("model.embed_tokens.weight", "model.layers.0.linear_attn.A_log", "model.norm.weight"):
            source_name = name.replace("model.", "model.language_model.", 1)
            if not torch.equal(base[name].detach().cpu(), tensors.get_tensor(source_name).to(base[name].dtype)):
                raise AssertionError(f"Loaded text weights differ from downloaded checkpoint: {name}")
            checked.append(source_name)
    with torch.no_grad():
        reference_before = frozen.batch_log_probabilities([chat], [codes])[0].cpu().numpy().copy()
    before = parameter_hash(actor)
    optimizer = torch.optim.AdamW([p for p in actor.model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0)
    optimizer.zero_grad(set_to_none=True)
    log_policy = actor.batch_log_probabilities([chat], [codes])[0]
    loss = -log_policy[0]  # Arbitrary action for a gradient/reload check, not task supervision.
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_([p for p in actor.model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
    optimizer.step()
    if float(norm) <= 0 or before == parameter_hash(actor):
        raise AssertionError("The downloaded model did not update its LoRA weights")
    actor.save_adapter(output / "discarded_adapter")
    with torch.no_grad():
        expected = actor.batch_log_probabilities([chat], [codes])[0].cpu().numpy().copy()
        reference_after = frozen.batch_log_probabilities([chat], [codes])[0].cpu().numpy().copy()
    if not np.array_equal(reference_before, reference_after):
        raise AssertionError("Actor training changed the independent frozen reference")
    metadata = actor.metadata
    del log_policy, loss, actor, optimizer
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    loaded = LanguagePolicy(str(args.model), device=args.device, trainable=False,
                            adapter_path=str(output / "discarded_adapter"), rank=16)
    with torch.no_grad():
        replay = loaded.batch_log_probabilities([chat], [codes])[0].cpu().numpy()
    error = float(np.max(np.abs(replay - expected)))
    if error > 1e-3:
        raise AssertionError(f"Saved adapter replay failed: {error}")
    return {"model": str(args.model), "reference_model": str(args.reference_model), "device": args.device,
            "gradient_norm": float(norm), "reload_max_log_error": error,
            "reference_max_log_drift": float(np.max(np.abs(reference_before - reference_after))),
            "verified_checkpoint_tensors": checked, "metadata": metadata}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-smoke", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model", type=Path, default=Path("models/Qwen3.5-0.8B"))
    parser.add_argument("--reference-model", type=Path, default=Path("models/Qwen3.5-2B"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    report = {"status": "running", "scope": "Discarded implementation smoke, not experimental results",
              "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (Path(__file__), ROOT / "environment/hrp_miniwob/adapters/meta_miniwob.py")}}
    try:
        report["browser_contracts"] = browser_check(args.output)
        if args.model_smoke:
            report["model_contract"] = model_check(args, args.output)
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        write_json(args.output / "status.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "model_contract"}), flush=True)


if __name__ == "__main__":
    main()
