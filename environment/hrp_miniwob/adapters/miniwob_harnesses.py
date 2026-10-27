"""Finite harness adapters over unchanged v2 pages, losses and graph metrics."""

from dataclasses import asdict, dataclass, replace
from functools import cached_property
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from hrp_miniwob.browser import Step
from hrp_miniwob.v2.browser import State as NativeState
from hrp_miniwob.v2.browser import Task as NativeTask
from hrp_miniwob.v2.model import compile_task

from hrp_miniwob.harnesses.miniwob_harnesses import CODES, Candidate
from hrp_miniwob.adapters import miniwob_v2 as base
from hrp_miniwob.adapters.miniwob_v2 import replay_start as replay_start, start_bank as start_bank
from hrp_miniwob.adapters.native_miniwob import write_json

MICROBATCH_STATES = base.MICROBATCH_STATES
exact_update = base.exact_update


@dataclass(frozen=True)
class Task(NativeTask):
    candidate: Candidate

    @property
    def native(self):
        return NativeTask(self.seed, self.family, self.instruction, self.data, self.initial)

    @cached_property
    def native_actions(self):
        return self.native.actions

    @cached_property
    def click_ids(self):
        return tuple(a.removeprefix("click:") for a in self.native_actions if a.startswith("click:"))

    @cached_property
    def checked_actions(self):
        return tuple(a for a in self.native_actions if self.native.checked_action(a))

    @cached_property
    def queues(self):
        return ((),) + tuple((a,) for a in self.click_ids) + tuple((a, b) for a in self.click_ids for b in self.click_ids)

    @cached_property
    def actions(self):
        return self.candidate.actions(self)

    @cached_property
    def memories(self):
        return self.candidate.memory_states(self)

    @property
    def fingerprint(self):
        # Pair tasks by public native contents, independent of deployment profile.
        return self.native.fingerprint


@dataclass(frozen=True)
class State(NativeState):
    def public(self):
        return super().public() | self.task.candidate.context(self.task, self.memory)


class Graph(base.Graph):
    def index(self, remaining, memory, physical):
        return ((remaining - 1) * self.task.memories + memory) * self.task.physical_states + physical

    def state(self, index):
        level, physical = divmod(index, self.task.physical_states)
        remaining, memory = divmod(level, self.task.memories)
        return State(self.task, physical, remaining + 1, memory)


class LanguagePolicy(base.LanguagePolicy):
    def __init__(self, *args, activation_checkpointing=True, **kwargs):
        super().__init__(*args, **kwargs)
        # Keep activation recomputation, BF16 and B=8 unchanged from E1.
        if activation_checkpointing and any(p.requires_grad for p in self.model.parameters()):
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.metadata["activation_checkpointing"] = "non-reentrant; unchanged fixed B=8 forward"
        for code in CODES:
            ids = self.tokenizer.encode(code, add_special_tokens=False)
            if len(ids) != 1 or self.tokenizer.decode(ids) != code:
                raise ValueError(f"Harness action code is not a reversible single token: {code}")
            self.code_token_ids[code] = ids[0]
        if len(set(self.code_token_ids.values())) != len(self.code_token_ids):
            raise ValueError("Harness action codes must have distinct tokens")
        self.metadata["harness_protocol"] = "h0_h5_v2"
        self.metadata["decoder"].update(native_codes=CODES, semantic_order="harness Task.actions")


class Adapter(base.Adapter):
    def graph(self, model, with_reference=True):
        key, candidate = model.task.key, self.candidate
        if key not in self.graphs:
            native = model.task.native if isinstance(model.task, Task) else model.task
            task = Task(**asdict(native), candidate=candidate)
            shape = model.horizon * task.memories * task.physical_states, len(task.actions)
            graph = Graph(task, model.horizon, candidate,
                          np.full(shape, -1, dtype=np.int64), np.zeros(shape))
            decisions = {}
            for index in range(shape[0]):
                state = graph.state(index)

                def check(action, mode, physical=state.physical):
                    review_key = physical, action, mode
                    if review_key not in decisions:
                        decisions[review_key] = self.review(native, physical, action, mode)
                    return decisions[review_key]

                for column, action in enumerate(task.actions):
                    memory, primitives, _ = candidate.request(task, state.memory, action, check)
                    physical, reward = state.physical, 0.
                    for primitive in primitives:
                        a = task.native_actions.index(primitive)
                        reward = float(model.physical_rewards[physical, a])
                        physical = int(model.physical_next[physical, a])
                        if physical < 0:
                            break
                    graph.task_rewards[index, column] = reward
                    if physical >= 0 and state.remaining > 1:
                        graph.next_state[index, column] = graph.index(state.remaining - 1, memory, physical)
            self.graphs[key] = graph
        # Reuse the v2 frozen-prior cache and fixed B=8 measurements, with the
        # harness's complete menu rather than the native page's smaller menu.
        graph = self.graphs[key]
        return super().graph(SimpleNamespace(task=graph.task), with_reference)


class Harness:
    def __init__(self, browser, adapter):
        self.browser, self.adapter, self.state = browser, adapter, None
        self.counts = dict.fromkeys(("actor_decisions", "browser_actions", "program_checks",
                                    "reviewer_calls", "dom_events", "batch_calls", "plan_writes", "plan_steps"), 0)
        self.last_review, self.last_execution = None, []

    def reset(self, task):
        native = task.native if isinstance(task, Task) else task
        state = self.browser.reset(native.seed, native.family)
        if state.task != native:
            raise AssertionError("Public task changed on reset")
        graph = self.adapter.graph(compile_task(native, self.browser.horizon), with_reference=False)
        self.state = State(graph.task, state.physical, state.remaining, 0)
        self.last_review, self.last_execution = None, []
        return self.state

    def step(self, action):
        state = self.state
        if state is None or state.terminated:
            raise RuntimeError("An active task is required")
        if action not in state.task.actions:
            raise ValueError("Unknown harness action")
        self.counts["actor_decisions"] += 1
        self.counts["batch_calls"] += int(action.startswith("batch:"))
        self.counts["plan_writes"] += int(action.startswith("plan:"))
        self.counts["plan_steps"] += int(action == "plan_next")

        def check(target, mode):
            self.counts["program_checks"] += int(mode == "program")
            self.counts["reviewer_calls"] += int(mode == "lm")
            return self.adapter.review(state.task.native, state.physical, target, mode)

        memory, primitives, self.last_review = self.adapter.candidate.request(state.task, state.memory, action, check)
        self.last_execution = []
        remaining, reward, raw = state.remaining - 1, 0., None
        physical, terminated = state.physical, False
        reason = "tool_result" if action.startswith("verify:") else "controller_action"
        if self.last_review and not self.last_review["approved"] and not action.startswith("verify:"):
            reason = "review_rejected"
        # Reserve enough internal browser steps to finish this ONE model
        # decision. Native success/failure still terminates immediately.
        self.browser.state = replace(self.browser.state, remaining=len(primitives) + 1)
        for primitive in primitives:
            before = self.browser.dom_events
            result = self.browser.step(primitive)
            self.counts["browser_actions"] += 1
            self.counts["dom_events"] += self.browser.dom_events - before
            self.last_execution.append({"action": primitive, "reward": result.reward,
                                        "raw_reward": result.raw_reward, "terminated": result.state.terminated})
            physical, terminated = result.state.physical, result.state.terminated
            reward, raw, reason = result.reward, result.raw_reward, result.reason
            if terminated:
                break
        if remaining == 0 and not terminated:
            self.browser.page.evaluate("core.endEpisode(0, false, 'decision budget exhausted')")
            terminated, raw, reason = True, 0., "budget"
        self.state = State(state.task, physical, remaining, memory, terminated)
        self.browser.state = replace(self.browser.state, remaining=remaining, terminated=terminated)
        return Step(self.state, reward, 0., raw, reason)


def evaluate_models(models, adapter, output):
    report = base.evaluate_models(models, adapter, output)
    report["harness"] = adapter.candidate.profile
    report["harness_protocol"] = "h0_h5_v2"
    report["action_sets"] = {}
    for model in models:
        graph = adapter.graph(model)
        report["action_sets"][model.task.key] = {"actions": list(graph.task.actions), "controller_states": graph.task.memories}
        path = Path(output) / "policies" / f"{model.task.key}.npz"
        with np.load(path) as saved:
            arrays = dict(saved)
        np.savez_compressed(path, **arrays, actions=np.asarray(graph.task.actions))
    write_json(Path(output) / "metrics.json", report)
    return report
