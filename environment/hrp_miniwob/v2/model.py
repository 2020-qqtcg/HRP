"""Finite kernels derived from public task contents, checked against native DOM.

Compiler rules do not access the page's private grading closures. The lexical
checker for the synonym family uses the pinned benchmark's public vocabulary.
"""

import ast
import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from ..assets import VENDOR
from .browser import State, Task


@lru_cache(maxsize=1)
def synonym_groups():
    source = (VENDOR / "miniwob/html/miniwob/click-checkboxes-soft.html").read_text()
    literal = re.search(r"var SYNONYMS\s*=\s*(\[.*?\]);", source, re.DOTALL).group(1)
    return {word: i for i, group in enumerate(ast.literal_eval(literal)) for word in group}


def target_mask(task):
    prefix = "Select words similar to " if task.family.endswith("-soft") else "Select "
    requested = task.instruction.removeprefix(prefix).removesuffix(" and click Submit.").split(", ")
    groups = synonym_groups() if task.family.endswith("-soft") else None
    return sum(1 << i for i, label in enumerate(task.data["labels"])
               if any(groups[label] == groups[word] if groups else label == word for word in requested))


def physical_edge(task, physical, action):
    """Return physical successor (-1 for termination), binary reward, native raw."""
    f, p = task.family, physical
    if f.startswith("click-checkboxes"):
        if action == "submit":
            n = len(task.data["labels"])
            raw = (n - 2 * (p ^ target_mask(task)).bit_count()) / n
            return -1, float(raw == 1.), raw
        return p ^ (1 << int(action.removeprefix("click:ch"))), 0., None
    number = task.data["tiles"][int(action.removeprefix("click:tile"))]["number"]
    if number <= p:
        return p, 0., None  # A click request to a removed tile is an explicit no-op.
    if number != p + 1:
        return -1, 0., -1. if p == 0 else p / 5
    return (p + 1, 0., None) if p < 4 else (-1, 1., 1.)


def public_approve(task, physical, action):
    successor, reward, _ = physical_edge(task, physical, action)
    return successor >= 0 or reward == 1.


def state_category(task, physical):
    if task.family.startswith("click-checkboxes"):
        target = target_mask(task)
        extra, missing = bool(physical & ~target), bool(target & ~physical)
        return ("mixed" if missing else "extra_only") if extra else ("missing_only" if missing else "correct")
    return f"progress_{physical}"


@dataclass
class TaskModel:
    task: Task
    horizon: int
    physical_next: np.ndarray
    physical_rewards: np.ndarray
    raw_rewards: np.ndarray
    next_state: np.ndarray
    task_rewards: np.ndarray
    browser_checks: int = 0

    @property
    def initial_state(self):
        return (self.horizon - 1) * self.task.physical_states + self.task.initial

    def state(self, index):
        level, physical = divmod(index, self.task.physical_states)
        return State(self.task, physical, level + 1)

    def reference(self):
        return np.full(self.next_state.shape, 1 / len(self.task.actions))


def compile_task(task, horizon=10):
    if horizon < 1:
        raise ValueError("Horizon must be positive")
    shape = task.physical_states, len(task.actions)
    physical, rewards, raw = np.full(shape, -1, dtype=np.int64), np.zeros(shape), np.full(shape, np.nan)
    for p in range(shape[0]):
        for a, action in enumerate(task.actions):
            successor, reward, native = physical_edge(task, p, action)
            physical[p, a], rewards[p, a] = successor, reward
            if native is not None:
                raw[p, a] = native
    graph = np.full((horizon, *shape), -1, dtype=np.int64)
    for level in range(1, horizon):
        graph[level] = np.where(physical >= 0, (level - 1) * shape[0] + physical, -1)
    return TaskModel(task, horizon, physical, rewards, raw, graph.reshape(-1, shape[1]), np.tile(rewards, (horizon, 1)))


def audit_model(browser, model, exhaustive=False):
    """Exercise every state's grading and a deterministic transition basis.

    exhaustive=True additionally executes every physical state/action pair.
    Both modes exercise ordinary and last-decision budget rules. Audit restore
    does not define the training start distribution; training requires prefixes.
    """
    checks = 0
    t = model.task
    basis = {t.initial, 0, t.physical_states - 1, t.physical_states // 2}
    for p in range(t.physical_states):
        for a, action in enumerate(t.actions):
            if not exhaustive and p not in basis and not t.checked_action(action):
                continue
            for remaining in (max(2, browser.horizon), 1):
                browser.restore(State(t, p, min(remaining, browser.horizon)))
                result = browser.step(action)
                successor = int(model.physical_next[p, a])
                terminal = successor < 0 or result.state.remaining == 0
                if result.state.terminated != terminal or result.reward != model.physical_rewards[p, a]:
                    raise AssertionError(f"Native grading/termination mismatch: {t.key}, {p}, {action}, {remaining}")
                if successor >= 0 and result.state.physical != successor:
                    raise AssertionError(f"Native physical transition mismatch: {t.key}, {p}, {action}")
                expected_raw = model.raw_rewards[p, a] if successor < 0 else 0. if terminal else None
                if expected_raw is not None and not np.isclose(result.raw_reward, expected_raw, rtol=0, atol=1e-14):
                    raise AssertionError(f"Native raw reward mismatch: {t.key}, {p}, {action}")
                checks += 1
    model.browser_checks += checks
    return checks
