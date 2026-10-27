"""Native MiniWoB agent loops with explicit roles and fixed optional reviewers."""

import json
from dataclasses import dataclass

from hrp_miniwob.browser import State, Step

HARNESSES = {
    "direct": {"guidance": "Choose the next action to complete the user's task.", "review": "none"},
    "careful": {"guidance": "Before choosing an action, compare every current checkbox with the user's request. "
                            "Submit only after checking that the selection matches exactly.", "review": "none"},
    "verify": {"guidance": "Choose the next action to complete the user's task.", "review": "program"},
    "reviewer": {"guidance": "Choose the next action to complete the user's task.", "review": "lm"},
    "reviewer_tool": {"guidance": "Choose the next action to complete the user's task.", "review": "lm_tool"},
}


def public_check(task, mask):
    # Only the displayed instruction and labels; no browser answer closure or DP.
    requested = task.instruction.removeprefix("Select ").removesuffix(" and click Submit.")
    words = set(requested.split(", ")) if requested and requested != "nothing" else set()
    selected = {label for i, label in enumerate(task.labels) if mask & (1 << i)}
    return selected == words


@dataclass(frozen=True)
class LoopState:
    task: object
    mask: int
    remaining: int
    check_slot: int = 1
    terminated: bool = False


def messages(state, name):
    spec = HARNESSES[name]
    workflow = {
        "none": "Every submission request commits the webpage submission and ends the task.",
        "program": "The first submission request is checked against the public instruction by a program. "
                   "If rejected, the page is kept for revision; subsequent submissions commit immediately.",
        "lm": "A frozen language-model reviewer without tools checks the first submission request. "
              "It may approve or reject and may be mistaken. Rejection leaves the page for revision; "
              "subsequent submissions commit immediately.",
        "lm_tool": "A frozen language-model reviewer receives a public-constraint check result when checking "
                   "the first submission request. It may approve or reject and may be mistaken. Rejection "
                   "leaves the page for revision; subsequent submissions commit immediately.",
    }[spec["review"]]
    system = (spec["guidance"] + " Each click toggles one checkbox. Every action request consumes one decision. "
              + workflow + " Only a committed, exactly correct submission earns reward 1. All other rewards "
              "and explicit costs are zero. Exhausting the decision budget ends the task. "
              "Optimize discounted reward with gamma=0.95 and KL regularization tau=0.1 to the fixed reference. "
              "Return exactly one listed action code.")
    codes = list("ABCDEFG"[:len(state.task.actions)])
    observation = {"instruction": state.task.instruction, "decisions_remaining": state.remaining,
                   "review_slot_remaining": state.check_slot,
                   "last_review": "rejected" if state.check_slot == 0 else "not_used",
                   "checkboxes": [{"id": f"ch{i}", "label": label, "checked": bool(state.mask & (1 << i))}
                                  for i, label in enumerate(state.task.labels)],
                   "actions": dict(zip(codes, state.task.actions, strict=True))}
    return [{"role": "system", "content": system},
            {"role": "user", "content": json.dumps(observation, ensure_ascii=False)}], codes


def review_messages(task, mask, with_tool):
    system = ("You are a frozen submission reviewer. Decide whether the current checkbox selection exactly "
              "satisfies the user's instruction. A means approve submission. B means reject for revision. "
              "Return exactly A or B. Do not propose or execute edits.")
    observation = {"instruction": task.instruction,
                   "checkboxes": [{"label": label, "checked": bool(mask & (1 << i))}
                                  for i, label in enumerate(task.labels)]}
    chat = [{"role": "system", "content": system},
            {"role": "user", "content": json.dumps(observation, ensure_ascii=False)}]
    if with_tool:
        # The fixed workflow invokes a real, deterministic program checker.
        chat += [{"role": "assistant", "content": "Check the public constraints.",
                  "tool_calls": [{"id": "public_check", "type": "function",
                                  "function": {"name": "check_public_constraints", "arguments": {}}}]},
                 {"role": "tool", "tool_call_id": "public_check", "name": "check_public_constraints",
                  "content": json.dumps({"pass": public_check(task, mask)})}]
    return chat


class BrowserHarness:
    """Intercept submission before irreversible native commit; never reset to retry."""

    def __init__(self, browser, name, reviewer):
        self.browser, self.name, self.reviewer = browser, name, reviewer
        self.state = None
        self.counts = {"actor_decisions": 0, "browser_actions": 0, "program_checks": 0, "reviewer_calls": 0}
        self.last_review = None

    def reset(self, seed):
        state = self.browser.reset(seed)
        self.state = LoopState(state.task, state.mask, state.remaining)
        return self.state

    def restore_for_audit(self, state):
        self.browser.restore(State(state.task, state.mask, state.remaining))
        self.state = state

    def step(self, action):
        state = self.state
        if state is None or state.terminated:
            raise RuntimeError("Cannot act on a terminated harness")
        if action not in state.task.actions:
            raise ValueError("Unknown semantic action")
        self.counts["actor_decisions"] += 1
        self.last_review = None
        mode = HARNESSES[self.name]["review"]
        if action == "submit" and state.check_slot and mode != "none":
            self.counts["program_checks"] += int(mode in {"program", "lm_tool"})
            self.counts["reviewer_calls"] += int(mode in {"lm", "lm_tool"})
            accepted = self.reviewer(state.task, state.mask)
            self.last_review = {"mode": mode, "accepted": accepted}
            if not accepted:
                remaining = state.remaining - 1
                if remaining == 0:
                    self.browser.page.evaluate("core.endEpisode(0, false, 'decision budget exhausted')")
                self.browser.state = State(state.task, state.mask, remaining, remaining == 0)
                self.state = LoopState(state.task, state.mask, remaining, 0, remaining == 0)
                return Step(self.state, 0., 0., 0. if remaining == 0 else None, "review_rejected")
        result = self.browser.step(action)
        self.counts["browser_actions"] += 1
        self.state = LoopState(result.state.task, result.state.mask, result.state.remaining,
                               state.check_slot, result.state.terminated)
        return Step(self.state, result.reward, result.cost, result.raw_reward, result.reason)
