"""Finite semantic decisions executed on the unmodified MiniWoB webpage."""

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone

from playwright.sync_api import sync_playwright

from .assets import ROOT, VENDOR, verify_assets


@dataclass(frozen=True)
class Task:
    seed: int
    instruction: str
    labels: tuple[str, ...]

    @property
    def actions(self) -> tuple[str, ...]:
        return tuple(f"click:ch{i}" for i in range(len(self.labels))) + ("submit",)

    @property
    def target_mask(self) -> int:
        """Parse only the user-visible instruction, never the JS answer closure."""
        requested = self.instruction.removeprefix("Select ").removesuffix(
            " and click Submit."
        ).split(", ")
        return sum(1 << i for i, label in enumerate(self.labels) if label in requested)


@dataclass(frozen=True)
class State:
    task: Task
    mask: int
    remaining: int
    terminated: bool = False

    def observation(self, style: str = "json") -> str:
        """Canonical, stateless model input; two encodings of identical information."""
        checkboxes = [
            {"id": f"ch{i}", "label": label, "checked": bool(self.mask & (1 << i))}
            for i, label in enumerate(self.task.labels)
        ]
        if style == "json":
            return json.dumps({
                "instruction": self.task.instruction,
                "remaining": self.remaining,
                "terminated": self.terminated,
                "checkboxes": checkboxes,
                "actions": [] if self.terminated else self.task.actions,
            }, ensure_ascii=False)
        if style != "text":
            raise ValueError("Observation style must be json or text")
        rows = [self.task.instruction, f"Decisions remaining: {self.remaining}",
                f"Terminated: {self.terminated}"]
        rows += [f'{x["id"]}: {x["label"]}; checked={x["checked"]}' for x in checkboxes]
        rows.append("Actions: " + ("none" if self.terminated else ", ".join(self.task.actions)))
        return "\n".join(rows)


@dataclass(frozen=True)
class Step:
    state: State
    reward: float
    cost: float
    raw_reward: float | None
    reason: str | None


class CheckboxBrowser:
    """Budgeted click-checkboxes task, using element clicks rather than coordinates.

    The page clock is paused before the episode starts. Native MiniWoB grading
    and the single-reward guard remain intact, but elapsed wall time is zero.
    Success earns 1, all other terminal outcomes earn 0, and costs are separate.
    """

    def __init__(self, horizon: int = 10, click_cost: float = 0.02,
                 submit_cost: float = 0.0, headed: bool = False):
        if horizon < 1:
            raise ValueError("horizon must be positive")
        verify_assets()
        self.horizon = horizon
        self.click_cost = click_cost
        self.submit_cost = submit_cost
        self.playwright = sync_playwright().start()
        # Ubuntu hosts without system Chromium libraries can use the local
        # install_native_deps.sh extraction; this changes only the child process.
        browser_env = os.environ.copy()
        libraries = str(ROOT / "vendor/native/root/usr/lib/x86_64-linux-gnu")
        existing_libraries = browser_env.get("LD_LIBRARY_PATH")
        browser_env["LD_LIBRARY_PATH"] = (
            libraries + os.pathsep + existing_libraries if existing_libraries else libraries
        )
        self.browser = self.playwright.chromium.launch(headless=not headed, env=browser_env)
        self.page = self.browser.new_page(viewport={"width": 640, "height": 480})
        self.page.clock.install(time=datetime(2025, 1, 1, tzinfo=timezone.utc))
        self.page.clock.pause_at(datetime(2025, 1, 1, 0, 0, 1, tzinfo=timezone.utc))
        self.page.goto((VENDOR / "miniwob/html/miniwob/click-checkboxes.html").as_uri())
        self.state: State | None = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self) -> None:
        self.browser.close()
        self.playwright.stop()

    def _mask(self) -> int:
        checked = self.page.locator('#boxes input[type="checkbox"]').evaluate_all(
            "elements => elements.map(x => x.checked)"
        )
        return sum(1 << i for i, value in enumerate(checked) if value)

    def reset(self, seed: int = 42) -> State:
        self.page.evaluate("seed => { Math.seedrandom(String(seed)); core.startEpisodeReal(); }", seed)
        task = Task(seed, self.page.locator("#query").inner_text(), tuple(
            self.page.locator("#boxes label").all_text_contents()
        ))
        self.state = State(task, self._mask(), self.horizon)
        return self.state

    def restore(self, state: State) -> State:
        """Rebuild a nonterminal snapshot by reset and real checkbox clicks.

        Restoration is an experiment/probe operation, not a policy action.
        The policy's remaining budget is restored after reconstruction.
        """
        if state.terminated or not 1 <= state.remaining <= self.horizon:
            raise ValueError("restore requires a nonterminal state within the budget")
        if not 0 <= state.mask < 2 ** len(state.task.labels):
            raise ValueError("mask exceeds the task's checkbox count")
        current = self.reset(state.task.seed)
        if current.task != state.task:
            raise ValueError("Restored task differs from the saved public task")
        for i in range(len(state.task.labels)):
            if state.mask & (1 << i):
                self.page.locator(f"#ch{i}").evaluate("element => element.click()")
        self.state = State(current.task, self._mask(), state.remaining)
        return self.state

    def step(self, action: str) -> Step:
        if self.state is None or self.state.terminated:
            raise RuntimeError("reset/restore a nonterminal state before acting")
        if action not in self.state.task.actions:
            raise ValueError(f"Unknown semantic action: {action}")
        task, remaining = self.state.task, self.state.remaining - 1
        selector = "#subbtn" if action == "submit" else "#" + action.removeprefix("click:")
        self.page.locator(selector).evaluate("element => element.click()")
        reason = "submit" if action == "submit" else None
        if remaining == 0 and action != "submit":
            self.page.evaluate("core.endEpisode(0, false, 'decision budget exhausted')")
            reason = "budget"
        metadata = self.page.evaluate("({done: WOB_DONE_GLOBAL, raw: WOB_RAW_REWARD_GLOBAL})")
        self.state = State(task, self._mask(), remaining, bool(metadata["done"]))
        raw = float(metadata["raw"]) if self.state.terminated else None
        reward = float(action == "submit" and raw == 1.0)
        cost = self.submit_cost if action == "submit" else self.click_cost
        return Step(self.state, reward, cost, raw, reason)

    def screenshot(self, path: str) -> None:
        self.page.screenshot(path=path, full_page=True)
