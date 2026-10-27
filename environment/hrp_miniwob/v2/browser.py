"""Public observations and real DOM actions on three unmodified upstream pages.

Ascending-number IDs are assigned by position, never by hidden answer attributes. Its initial
visible layout is retained as observation memory, making the task Markov.
"""

import hashlib
import json
from dataclasses import asdict, dataclass

from ..assets import VENDOR
from ..browser import CheckboxBrowser, Step
from . import FAMILIES
from .assets import verify_assets


@dataclass(frozen=True)
class Task:
    seed: int
    family: str
    instruction: str
    data: dict
    initial: int

    def __post_init__(self):
        if self.family not in FAMILIES:
            raise ValueError(f"Unknown v2 task family: {self.family}")

    @property
    def key(self):
        return f"{self.family}_{self.seed}"

    @property
    def fingerprint(self):
        public = asdict(self)
        del public["seed"]
        return hashlib.sha256(json.dumps(public, sort_keys=True).encode()).hexdigest()

    @property
    def actions(self):
        if self.family.startswith("click-checkboxes"):
            return tuple(f"click:ch{i}" for i in range(len(self.data["labels"]))) + ("submit",)
        return tuple(f"click:tile{i}" for i in range(5))

    @property
    def physical_states(self):
        if self.family.startswith("click-checkboxes"):
            return 2 ** len(self.data["labels"])
        return 5

    def checked_action(self, action):
        return action == "submit" or self.family == "ascending-numbers"


@dataclass(frozen=True)
class State:
    task: Task
    physical: int
    remaining: int
    memory: int = 0
    terminated: bool = False

    def public(self):
        t, p = self.task, self.physical
        view = {"family": t.family, "instruction": t.instruction,
                "decisions_remaining": self.remaining, "controller_memory": self.memory}
        if t.family.startswith("click-checkboxes"):
            view["checkboxes"] = [{"id": f"ch{i}", "label": label, "checked": bool(p & (1 << i))}
                                  for i, label in enumerate(t.data["labels"])]
        else:
            view["initial_visible_layout"] = t.data["tiles"]
            offset = t.data["hidden_box_offset"] if p else {"x": 0, "y": 0}
            view["visible_tiles"] = [{"id": tile["id"], "x": tile["x"] + offset["x"], "y": tile["y"] + offset["y"],
                                      "text": str(tile["number"]) if p == 0 else None}
                                     for tile in t.data["tiles"] if tile["number"] > p]
            view["completed_clicks"] = p
        return view


class Browser(CheckboxBrowser):
    def __init__(self, horizon=10, click_cost=0., submit_cost=0., headed=False):
        if click_cost or submit_cost:
            raise ValueError("v2 uses zero explicit costs")
        verify_assets()
        super().__init__(horizon, click_cost, submit_cost, headed)
        self.dom_events = 0

    def reset(self, seed=42, family="click-checkboxes"):
        if family not in FAMILIES:
            raise ValueError(f"Unknown v2 task family: {family}")
        # Fresh document also resets native widget state (including dirty inputs).
        self.page.goto((VENDOR / f"miniwob/html/miniwob/{family}.html").as_uri())
        self.page.evaluate("seed => { Math.seedrandom(String(seed)); core.startEpisodeReal(); }", seed)
        data = {}
        if family.startswith("click-checkboxes"):
            data["labels"] = self.page.locator("#boxes label").all_text_contents()
        elif family == "ascending-numbers":
            tiles = self.page.locator("svg text").evaluate_all("es => es.map(e => {const b=e.getBoundingClientRect(); return {number:Number(e.textContent),x:b.left,y:b.top};})")
            data["tiles"] = [dict(tile, id=f"tile{i}") for i, tile in enumerate(sorted(tiles, key=lambda t: (t["x"], t["y"])))]
            # The native page computes rect x=old_screen_x-5, y=old_screen_y-50
            # inside its SVG. Retain this public coordinate transformation.
            box = self.page.locator("svg").bounding_box()
            data["hidden_box_offset"] = {"x": box["x"] - 5, "y": box["y"] - 50}
        task = Task(seed, family, self.page.locator("#query").inner_text(), data, 0)
        initial = self._physical(task)
        task = Task(seed, family, task.instruction, data, initial)
        self.state = State(task, initial, self.horizon)
        return self.state

    def _physical(self, task):
        if task.family.startswith("click-checkboxes"):
            return self._mask()
        return 0 if self.page.locator("svg text").count() else 5 - self.page.locator("svg rect").count()

    def _execute(self, action):
        task = self.state.task
        if task.family == "ascending-numbers":
            tile = task.data["tiles"][int(action.removeprefix("click:tile"))]
            # Select only by the public screen position, never data-index/order.
            clicked = self.page.locator("svg text, svg rect").evaluate_all("(es, arg) => {const e=es.find(e => {const b=e.getBoundingClientRect(), off=e.tagName==='rect' ? arg.offset : {x:0,y:0}; return Math.abs(b.left-arg.pos.x-off.x)<0.1 && Math.abs(b.top-arg.pos.y-off.y)<0.1;}); if(e) e.dispatchEvent(new MouseEvent('click', {bubbles:true})); return Boolean(e);}", {"pos": tile, "offset": task.data["hidden_box_offset"]})
            self.dom_events += int(clicked)
        else:
            selector = "#subbtn" if action == "submit" else "#" + action.removeprefix("click:")
            self.page.locator(selector).evaluate("e => e.click()")
            self.dom_events += 1

    def step(self, action):
        state = self.state
        if state is None or state.terminated:
            raise RuntimeError("Reset a nonterminal task before acting")
        if action not in state.task.actions:
            raise ValueError(f"Unknown action: {action}")
        self._execute(action)
        metadata = self.page.evaluate("({done: WOB_DONE_GLOBAL, raw: WOB_RAW_REWARD_GLOBAL})")
        raw = float(metadata["raw"]) if metadata["done"] else None
        reward = float(metadata["done"] and raw == 1.)
        remaining = state.remaining - 1
        reason = "native_terminal" if metadata["done"] else None
        if remaining == 0 and not metadata["done"]:
            self.page.evaluate("core.endEpisode(0, false, 'decision budget exhausted')")
            metadata["done"], raw, reason = True, 0., "budget"
        self.state = State(state.task, self._physical(state.task), remaining, state.memory, bool(metadata["done"]))
        return Step(self.state, reward, 0., raw, reason)

    def restore(self, state):
        """Audit only: reconstruct a physical snapshot using ordinary UI actions.

        Training coverage uses legal full graph prefixes, not this operation.
        """
        if state.terminated or not 1 <= state.remaining <= self.horizon:
            raise ValueError("Restore requires a live in-budget state")
        current = self.reset(state.task.seed, state.task.family)
        if current.task != state.task:
            raise AssertionError("Seeded public task changed")
        t, p = state.task, state.physical
        if not 0 <= p < t.physical_states:
            raise ValueError("Physical state outside task domain")
        if t.family.startswith("click-checkboxes"):
            prefix = [a for i, a in enumerate(t.actions[:-1]) if p & (1 << i)]
        else:
            prefix = ["click:" + next(tile["id"] for tile in t.data["tiles"] if tile["number"] == n) for n in range(1, p + 1)]
        for action in prefix:
            self._execute(action)
        if self._physical(t) != p or self.page.evaluate("WOB_DONE_GLOBAL"):
            raise AssertionError("UI reconstruction disagrees with requested nonterminal state")
        self.state = state
        return state
