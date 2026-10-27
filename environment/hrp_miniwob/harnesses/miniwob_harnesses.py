"""Finite MiniWoB agent interfaces; no free-text memory or answer planner."""

import ast
import hashlib
import json
import string
from pathlib import Path

CODES = string.ascii_uppercase + string.ascii_lowercase
PROFILES = {
    "H0": "Execute one native action per decision.",
    "H1": "You may call verify on a listed action. A public program returns whether it avoids an incorrect terminal outcome. Verification does not execute or block the action. Its result is cleared by any native action.",
    "H2": "Commit actions are automatically checked by a public program before execution. Rejection preserves the page and consumes this decision. After the first rejection all subsequent actions execute without review.",
    "H3": "You may batch two clicks in the specified order. There is no model decision between them. Native termination stops the batch immediately. The entire batch consumes one decision; each attempted native click is counted separately.",
    "H4": "You may call verify on a listed action. An independent frozen language-model reviewer returns whether it avoids an incorrect terminal outcome and may be mistaken. Verification does not execute or block the action. Its result is cleared by any native action.",
    "H5": (
        "Execute one native action per decision. Use this procedure on every observation. "
        "For click-checkboxes, identify the complete set of labels requested by the instruction. "
        "For click-checkboxes-soft, match labels by meaning, including synonyms, rather than "
        "requiring identical spelling. Compare each checkbox's current checked status with "
        "whether it belongs in that set. Click one mismatched checkbox: either clear an unwanted "
        "selection or select a missing requested label. Leave already correct checkboxes alone. "
        "Recheck the updated observation after each click. Submit when every checkbox matches, "
        "and reserve one decision for Submit. For ascending-numbers, the next required number "
        "is completed_clicks + 1. Find its position ID in initial_visible_layout and confirm "
        "that ID is still in visible_tiles. Use that ID even when the number text is hidden. "
        "Tile IDs describe positions, not numeric order; clicking a removed tile wastes a "
        "decision. Finally, look up the chosen native action in this observation's action menu "
        "and return its exact case-sensitive code. Return the code alone, without an ID, "
        "explanation or additional action."
    ),
}


class Candidate:
    def __init__(self, path):
        self.path = Path(path).resolve()
        source = self.path.read_text()
        statements = ast.parse(source).body
        if statements and isinstance(statements[0], ast.Expr) and isinstance(statements[0].value, ast.Constant):
            statements = statements[1:]
        if (len(statements) != 1 or not isinstance(statements[0], ast.Assign)
                or len(statements[0].targets) != 1
                or not isinstance(statements[0].targets[0], ast.Name)
                or statements[0].targets[0].id != "HARNESS"):
            raise ValueError("A candidate must declare exactly HARNESS = one of: " + ", ".join(PROFILES))
        self.profile = ast.literal_eval(statements[0].value)
        if self.profile not in PROFILES:
            raise ValueError("Unknown harness profile")
        self.sha256 = hashlib.sha256(source.encode()).hexdigest()
        self.initial_memory = 0

    def actions(self, task):
        actions = task.native_actions
        if self.profile in ("H1", "H4"):
            actions += tuple(f"verify:{action}" for action in task.checked_actions)
        elif self.profile == "H3":
            actions += tuple(f"batch:{a},{b}" for a in task.click_ids for b in task.click_ids)
        if len(actions) > len(CODES):
            raise ValueError("Action menu exceeds the verified single-token code alphabet")
        return actions

    def memory_states(self, task):
        if self.profile in ("H1", "H4"):
            return 1 + 2 * len(task.checked_actions)
        if self.profile == "H2":
            return 2
        return 1

    def context(self, task, memory):
        if self.profile in ("H1", "H4"):
            result = None
            if memory:
                target, approved = divmod(memory - 1, 2)
                result = {"action": task.checked_actions[target], "approved": bool(approved)}
            return {"verify_result": result}
        if self.profile == "H2":
            return {"automatic_review_enabled": memory == 0}
        return {}

    def request(self, task, memory, action, check):
        """Return new controller state, native calls and optional check result.

        check(action, mode) reads the public page or the fixed reviewer. It never
        receives DP values, actor weights or a desired next-action label.
        """
        if action.startswith("verify:"):
            target = action.removeprefix("verify:")
            mode = "program" if self.profile == "H1" else "lm"
            approved = bool(check(target, mode))
            memory = 1 + 2 * task.checked_actions.index(target) + int(approved)
            return memory, (), {"mode": mode, "action": target, "approved": approved}
        if action.startswith("batch:"):
            return 0, tuple("click:" + item for item in action.removeprefix("batch:").split(",")), None
        if self.profile == "H2" and memory == 0 and action in task.checked_actions:
            approved = bool(check(action, "program"))
            result = {"mode": "program", "action": action, "approved": approved}
            return (0, (action,), result) if approved else (1, (), result)
        return (memory if self.profile == "H2" else 0), (action,), None

    def messages(self, state):
        observation = state.public()
        codes = list(CODES[:len(state.task.actions)])
        observation["actions"] = dict(zip(codes, state.task.actions, strict=True))
        system = (
            "Operate the native MiniWoB page using one listed action code. Read every requested "
            "condition. Compare it with the current controls. Correct wrong selections and fill "
            "missing ones. Before a terminal action, check all conditions. For disappearing numbers, "
            "use the public initial layout and position IDs. Checkbox clicks toggle. A click on a "
            "removed tile is a no-op. Each selected action, including a tool or plan operation, "
            "consumes one decision. Budget exhaustion terminates. Only exact native completion "
            "earns 1; other rewards and explicit costs are zero. Optimize discounted reward "
            "gamma=0.95 and KL regularization tau=0.1 to the independent frozen reference. "
            "Public program checks use the instruction, observed page and fixed synonym vocabulary. "
            + PROFILES[self.profile] + " Return exactly one listed action code."
        )
        return [{"role": "system", "content": system},
                {"role": "user", "content": json.dumps(observation, ensure_ascii=False)}], codes
