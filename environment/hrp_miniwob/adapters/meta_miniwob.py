"""Finite, code-generated MiniWoB harnesses with independent frozen models.

Candidate code formats public observations and controls submission checks.
Only the evaluated actor selects semantic click/submit actions. This restricted
Python namespace is an experiment contract, not an OS security sandbox.
"""

import ast
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from hrp_miniwob.browser import State, Step
from hrp_miniwob.measurements import policy_gradient_weights, policy_metrics
from hrp_miniwob.suite import reachable_states

from hrp_miniwob.harnesses.miniwob_controlled import public_check, review_messages
from hrp_miniwob.adapters.native_miniwob import FrozenDecoder, write_json

MODES = ("none", "program", "lm", "lm_tool")
CONTEXT_TOKENS = 2048


def check_context(policy, chat):
    tokens = policy.tokenizer(policy.chat_text(chat), add_special_tokens=False, truncation=False)["input_ids"]
    if len(tokens) > CONTEXT_TOKENS:
        raise ValueError(f"Full templated prompt has {len(tokens)} tokens; actor/reference limit is {CONTEXT_TOKENS}")


class Candidate:
    def __init__(self, path):
        self.path = Path(path).resolve()
        code = self.path.read_text()
        self.sha256 = hashlib.sha256(code.encode()).hexdigest()
        tree = ast.parse(code, filename=str(self.path))

        def immutable(value):
            return type(value) in (str, int, float, bool, type(None)) or (
                type(value) is tuple and all(immutable(item) for item in value))

        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                if node.decorator_list or any(
                    not immutable(ast.literal_eval(value))
                    for value in [*node.args.defaults, *(d for d in node.args.kw_defaults if d is not None)]
                ):
                    raise ValueError("Candidate functions cannot retain mutable defaults or use decorators")
            elif isinstance(node, ast.Assign):
                if not all(isinstance(target, ast.Name) for target in node.targets) or not immutable(ast.literal_eval(node.value)):
                    raise ValueError("Module constants must be immutable literals; all memory belongs in controller state")
            elif not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                      and isinstance(node.value.value, str)):
                raise TypeError("Candidate modules may contain only functions, immutable constants and docstrings")
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal,
                                 ast.ClassDef, ast.AsyncFunctionDef, ast.While)):
                raise TypeError("Candidates must use pure functions without imports, global mutation or while loops")
            if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
                raise ValueError("Private attributes are outside the candidate contract")
            if isinstance(node, ast.Attribute) and isinstance(node.ctx, (ast.Store, ast.Del)):
                raise TypeError("Candidates cannot mutate module or function attributes")
            if isinstance(node, ast.Name) and node.id.startswith("__"):
                raise ValueError("Dunder names are outside the candidate contract")
        namespace = {"__builtins__": {
            "str": str, "int": int, "bool": bool, "len": len, "range": range,
            "enumerate": enumerate, "zip": zip, "list": list, "dict": dict,
            "tuple": tuple, "sorted": sorted, "reversed": reversed,
            "min": min, "max": max, "sum": sum, "any": any, "all": all,
        }, "json": SimpleNamespace(dumps=json.dumps, loads=json.loads)}
        exec(compile(tree, str(self.path), "exec"), namespace)  # noqa: S102 - execute the explicit candidate-code contract
        self.memories = namespace["MEMORY_STATES"]
        self.initial_memory = namespace["INITIAL_MEMORY"]
        if type(self.memories) is not int or not 1 <= self.memories <= 4:
            raise ValueError("MEMORY_STATES must be an integer in [1, 4]")
        if type(self.initial_memory) is not int or not 0 <= self.initial_memory < self.memories:
            raise ValueError("INITIAL_MEMORY is outside the controller state space")
        self.render = namespace["render"]
        self._review = namespace["review_mode"]
        self._advance = namespace["advance"]
        # The controller receives no task contents: it cannot choose the answer.
        for memory in range(self.memories):
            self.review_mode(memory)
            for action in [*(f"click:ch{i}" for i in range(6)), "submit"]:
                for approved in ((False, True) if action == "submit" else (True,)):
                    self.advance(memory, action, approved)

    def review_mode(self, memory):
        mode = self._review(memory)
        if mode not in MODES or mode != self._review(memory):
            raise ValueError("review_mode must deterministically return a supported mode")
        return mode

    def advance(self, memory, action, approved):
        result = self._advance(memory, action, approved)
        if (type(result) is not int or not 0 <= result < self.memories
                or result != self._advance(memory, action, approved)):
            raise ValueError("advance must deterministically return a valid memory state")
        return result

    def messages(self, state):
        codes = list("ABCDEFG"[:len(state.task.actions)])
        observation = {
            "instruction": state.task.instruction,
            "checkboxes": [{"id": f"ch{i}", "label": label,
                            "checked": bool(state.mask & (1 << i))}
                           for i, label in enumerate(state.task.labels)],
            "decisions_remaining": state.remaining, "controller_memory": state.memory,
            "submission_check": self.review_mode(state.memory),
            "actions": dict(zip(codes, state.task.actions, strict=True)),
        }
        # Fresh copies prevent the renderer from altering the authoritative view.
        extra = self.render(json.loads(json.dumps(observation)))
        if not isinstance(extra, str) or extra != self.render(json.loads(json.dumps(observation))):
            raise ValueError("render must return a deterministic string")
        if len(extra) > 4000:
            raise ValueError("Candidate rendering exceeds 4000 characters")
        system = (
            "Each click toggles exactly one checkbox. Each action request consumes one decision. "
            "Submission with check 'none' commits immediately. Otherwise the named fixed checker "
            "approves or rejects submission; rejection keeps the current page for correction. "
            "The program checker uses the public instruction; the frozen LM checker may err. "
            "Only a committed, exactly correct submission earns 1. All other rewards and explicit "
            "costs are zero. Budget exhaustion terminates. Optimize discounted reward with "
            "gamma=0.95 and KL regularization tau=0.1 to the fixed reference. "
            "Return exactly one listed action code.\n" + extra
        )
        return [{"role": "system", "content": system},
                {"role": "user", "content": json.dumps(observation, ensure_ascii=False)}], codes


@dataclass(frozen=True)
class MetaState:
    task: object
    mask: int
    remaining: int
    memory: int
    terminated: bool = False


@dataclass
class MetaTask:
    task: object
    horizon: int
    candidate: Candidate
    next_state: np.ndarray
    task_rewards: np.ndarray
    reference_table: np.ndarray | None = None

    @property
    def masks(self):
        return 2 ** len(self.task.labels)

    @property
    def initial_state(self):
        return self.index(self.horizon, self.candidate.initial_memory, 0)

    def index(self, remaining, memory, mask):
        return ((remaining - 1) * self.candidate.memories + memory) * self.masks + mask

    def state(self, index):
        level, mask = divmod(index, self.masks)
        remaining, memory = divmod(level, self.candidate.memories)
        return MetaState(self.task, mask, remaining + 1, memory)

    def reference(self):
        if self.reference_table is None:
            raise RuntimeError("Measure the independent frozen reference before DP")
        return self.reference_table


class MetaBrowser:
    def __init__(self, browser, adapter):
        self.browser, self.adapter = browser, adapter
        self.state = None
        self.counts = dict.fromkeys(("actor_decisions", "browser_actions", "program_checks", "reviewer_calls"), 0)
        self.last_review = None

    def reset(self, seed):
        state = self.browser.reset(seed)
        self.state = MetaState(state.task, state.mask, state.remaining, self.adapter.candidate.initial_memory)
        return self.state

    def restore_for_audit(self, state):
        self.browser.restore(State(state.task, state.mask, state.remaining, state.terminated))
        self.state = state

    def step(self, action):
        state, candidate = self.state, self.adapter.candidate
        if state is None or state.terminated:
            raise RuntimeError("An active task is required")
        if action not in state.task.actions:
            raise ValueError("Unknown semantic action")
        self.counts["actor_decisions"] += 1
        self.last_review = None
        approved = True
        if action == "submit":
            mode = candidate.review_mode(state.memory)
            self.counts["program_checks"] += int(mode in ("program", "lm_tool"))
            self.counts["reviewer_calls"] += int(mode in ("lm", "lm_tool"))
            approved = self.adapter.review(state.task, state.mask, mode)
            self.last_review = {"mode": mode, "approved": approved}
        memory = candidate.advance(state.memory, action, approved)
        if not approved:
            remaining = state.remaining - 1
            if remaining == 0:
                self.browser.page.evaluate("core.endEpisode(0, false, 'decision budget exhausted')")
            self.browser.state = State(state.task, state.mask, remaining, remaining == 0)
            self.state = replace(state, remaining=remaining, memory=memory, terminated=remaining == 0)
            return Step(self.state, 0., 0., 0. if remaining == 0 else None, "review_rejected")
        result = self.browser.step(action)
        self.counts["browser_actions"] += 1
        self.state = MetaState(result.state.task, result.state.mask, result.state.remaining,
                               memory, result.state.terminated)
        return Step(self.state, result.reward, result.cost, result.raw_reward, result.reason)


class MetaAdapter:
    def __init__(self, policy, candidate, reference_policy, reviewer_policy, cache):
        self.policy, self.candidate = policy, candidate
        cache = Path(cache)
        self.frozen = FrozenDecoder(reference_policy, cache / "reference.sqlite")
        self.reviewer = FrozenDecoder(reviewer_policy, cache / "reviewer.sqlite")
        self.graphs = {}
        self.checked_prompts = set()

    def close(self):
        self.frozen.db.close()
        self.reviewer.db.close()

    def messages(self, state):
        chat, codes = self.candidate.messages(state)
        key = json.dumps(chat, sort_keys=True)
        if key not in self.checked_prompts:
            check_context(self.policy, chat)
            check_context(self.frozen.policy, chat)
            self.checked_prompts.add(key)
        return chat, codes

    def review(self, task, mask, mode):
        if mode == "none":
            return True
        if mode == "program":
            return public_check(task, mask)
        chat = review_messages(task, mask, mode == "lm_tool")
        check_context(self.reviewer.policy, chat)
        lp = self.reviewer(chat, ["A", "B"])
        return bool(np.argmax(lp) == 0)

    def graph(self, model):
        if model.task.seed in self.graphs:
            return self.graphs[model.task.seed]
        c = self.candidate
        shape = (model.horizon * c.memories * len(model.physical_next), len(model.task.actions))
        graph = MetaTask(model.task, model.horizon, c, np.full(shape, -1, dtype=np.int64), np.zeros(shape))
        checked = {(mode, mask): self.review(model.task, mask, mode)
                   for mode in {c.review_mode(m) for m in range(c.memories)} for mask in range(graph.masks)}
        for index in range(shape[0]):
            state = graph.state(index)
            for action_index, action in enumerate(model.task.actions):
                approved = action != "submit" or checked[c.review_mode(state.memory), state.mask]
                memory = c.advance(state.memory, action, approved)
                if action == "submit" and approved:
                    graph.task_rewards[index, action_index] = float(model.raw_submit_rewards[state.mask] == 1.)
                elif state.remaining > 1:
                    mask = state.mask if action == "submit" else int(model.physical_next[state.mask, action_index])
                    graph.next_state[index, action_index] = graph.index(state.remaining - 1, memory, mask)
        # Unreachable rows have zero occupancy; a positive prior completes the table.
        graph.reference_table = np.full(shape, 1 / shape[1])
        for index in reachable_states(graph):
            chat, codes = self.messages(graph.state(int(index)))
            graph.reference_table[index] = np.exp(self.frozen(chat, codes))
        self.graphs[model.task.seed] = graph
        return graph

    def measure(self, model):
        graph = self.graph(model)
        lp = np.log(graph.reference()).copy()
        with torch.no_grad():
            for index in reachable_states(graph):
                chat, codes = self.messages(graph.state(int(index)))
                lp[index] = self.policy.batch_log_probabilities([chat], [codes])[0].detach().cpu().numpy()
        return graph, lp


def evaluate_models(models, adapter, output):
    output = Path(output)
    (output / "policies").mkdir(parents=True, exist_ok=True)
    rows = []
    for model in models:
        graph, lp = adapter.measure(model)
        row = policy_metrics(graph, np.exp(lp), np.zeros(len(graph.task.actions)), graph.reference())
        if row["gap_identity_error"] > 1e-8:
            raise AssertionError("Independent occupancy/KL mismatch disagrees with C-J")
        row.update(task_seed=model.task.seed, n=len(model.task.labels), k=model.task.target_mask.bit_count())
        rows.append(row)
        np.savez_compressed(output / "policies" / f"{model.task.seed}.npz", log_policy=lp,
                            reference=graph.reference(), next_state=graph.next_state, rewards=graph.task_rewards)
        write_json(output / "progress.json", {"tasks": len(rows), "total": len(models)})
    if not rows:
        raise ValueError("Evaluation requires a nonempty task split")
    metrics = {key: float(np.mean([row[key] for row in rows]))
               for key in rows[0] if key not in ("task_seed", "n", "k")}
    report = {"tasks": len(rows), "metrics": metrics, "rows": rows,
              "candidate_sha256": adapter.candidate.sha256,
              "backend": "browser-verified finite controller; independent frozen reference; exact occupancy/KL"}
    write_json(output / "metrics.json", report)
    return report


def exact_update(actor, model, adapter):
    graph, lp = adapter.measure(model)
    weights, advantage = policy_gradient_weights(graph, np.exp(lp), np.zeros(len(graph.task.actions)), graph.reference())
    actor.engine.optimizer_zero_grad()
    error = 0.
    bank = reachable_states(graph)
    for index in bank:
        chat, codes = adapter.messages(graph.state(int(index)))
        current = actor.engine.policy.batch_log_probabilities([chat], [codes])[0]
        error = max(error, float(np.max(np.abs(current.detach().cpu().numpy() - lp[index]))))
        loss = -weights[index] * (current.exp() * current.new_tensor(advantage[index])).sum()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite exact PG loss")
        loss.backward()
    if error > 1e-3:
        raise AssertionError(f"Policy replay failed: {error}")
    norm = actor.engine.optimizer_step()
    actor.engine.lr_scheduler_step()
    return {"task_seed": model.task.seed, "states": len(bank), "grad_norm": float(norm),
            "replay_max_log_error": error, "actor_backward_observations": len(bank)}
