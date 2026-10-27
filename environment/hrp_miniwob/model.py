"""Compile browser-observed edges into a finite budgeted task graph."""

from dataclasses import dataclass

import numpy as np

from .browser import CheckboxBrowser, State, Task


@dataclass(frozen=True)
class TaskModel:
    task: Task
    horizon: int
    next_state: np.ndarray
    task_rewards: np.ndarray
    physical_next: np.ndarray
    raw_submit_rewards: np.ndarray
    browser_checks: int

    @property
    def initial_state(self) -> int:
        return (self.horizon - 1) * 2 ** len(self.task.labels)

    def state(self, index: int) -> State:
        remaining, mask = divmod(index, 2 ** len(self.task.labels))
        return State(self.task, mask, remaining + 1)

    def costs(self, click_cost: float, submit_cost: float) -> np.ndarray:
        return np.array([click_cost] * len(self.task.labels) + [submit_cost])

    def reference(self, submit_probability: float | None = None) -> np.ndarray:
        n = len(self.task.labels)
        p = 1 / (n + 1) if submit_probability is None else submit_probability
        if not 0 < p < 1:
            raise ValueError("submit reference probability must be strictly between 0 and 1")
        return np.tile([(1 - p) / n] * n + [p], (len(self.next_state), 1))


def compile_task(browser: CheckboxBrowser, seed: int = 42) -> TaskModel:
    """Enumerate ALL checkbox masks/actions with real browser execution.

    The reward and physical transition tables are read from the browser.
    Independent XOR/public-instruction checks certify this task's abstraction.
    Budget lifting adds only the declared remaining-step counter and terminal
    rule. Both ordinary transitions and last-step transitions are exercised.
    """
    initial = browser.reset(seed)
    task, horizon = initial.task, browser.horizon
    if initial.mask != 0:
        raise AssertionError("Upstream initial checkbox state is not empty")
    n = len(task.labels)
    masks, actions = 2 ** n, n + 1
    physical_next = np.full((masks, actions), -1, dtype=np.int64)
    raw_rewards = np.empty(masks)
    checks = 0
    for mask in range(masks):
        for action_index, action in enumerate(task.actions):
            browser.restore(State(task, mask, horizon))
            result = browser.step(action)
            checks += 1
            if action == "submit":
                raw_rewards[mask] = result.raw_reward
                expected_raw = (n - 2 * (mask ^ task.target_mask).bit_count()) / n
                if result.raw_reward != expected_raw:
                    raise AssertionError("Browser grading disagrees with the public instruction")
                if not result.state.terminated or result.reward != float(mask == task.target_mask):
                    raise AssertionError("Submit must terminate and grade the actual checkbox state")
            else:
                physical_next[mask, action_index] = result.state.mask
                if result.state.mask != mask ^ (1 << action_index):
                    raise AssertionError("A semantic click does not toggle the requested checkbox")
                if result.reward != 0 or result.state.terminated != (horizon == 1):
                    raise AssertionError("A checkbox click has an unexpected reward/termination")

            # Same physical state at the final decision: termination is part of
            # the task, including the case where a click completes the right mask.
            browser.restore(State(task, mask, 1))
            last = browser.step(action)
            checks += 1
            if not last.state.terminated or last.state.remaining != 0:
                raise AssertionError("The last decision must terminate")
            if last.state.mask != result.state.mask:
                raise AssertionError("Physical transitions depend on the remaining budget")
            if last.reward != result.reward:
                raise AssertionError("Final-step reward differs from the declared task reward")
            try:
                browser.step("submit")
            except RuntimeError:
                pass
            else:
                raise AssertionError("A terminated task accepted another rewarded action")

    next_state = np.full((horizon * masks, actions), -1, dtype=np.int64)
    task_rewards = np.zeros_like(next_state, dtype=float)
    for remaining in range(1, horizon + 1):
        start = (remaining - 1) * masks
        if remaining > 1:
            next_state[start:start + masks, :n] = (
                (remaining - 2) * masks + physical_next[:, :n]
            )
        task_rewards[start:start + masks, n] = raw_rewards == 1.0
    return TaskModel(task, horizon, next_state, task_rewards, physical_next, raw_rewards, checks)
