"""Offline common-action projections for cached native HRP probes.

The native adapters keep the complete invocation.  This module adds a small,
versioned comparison projection without changing those records or claiming
cross-harness execution equivalence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from src.adapter.terminal_actions import analyze_bash_command


PROJECTION_VERSION = "native-common-action-v1"
COMMON_ACTIONS = (
    "inspect", "modify", "verify", "setup", "run", "plan", "delegate",
    "wait", "stop", "context_update", "return", "opaque", "invalid",
)
_ACTION_PRIORITY = {
    "opaque": 0,
    "wait": 1,
    "inspect": 2,
    "plan": 3,
    "delegate": 4,
    "run": 5,
    "verify": 6,
    "setup": 7,
    "modify": 8,
    "stop": 9,
}
_CONTROL_NAMES = {"completion_flag"}
_SHELL_TOOLS = {"bash", "execute"}
_PLAN_TOOLS = {"write_todos", "todo_write"}
_DELEGATE_TOOLS = {"task", "subagent"}
_INSPECT_OPERATIONS = {"read", "list", "search"}
_MODIFY_OPERATIONS = {"write", "edit", "delete"}


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def _dominant(categories: Iterable[str]) -> str:
    values = [x for x in categories if x in _ACTION_PRIORITY]
    return max(values, key=lambda x: _ACTION_PRIORITY[x], default="opaque")


def _message_texts(request: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    for message in request.get("messages", []) if isinstance(request, dict) else []:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            texts.extend(
                part.get("text", "") for part in content
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            )
    return texts


def _shell_context(row: dict[str, Any]) -> bool:
    """Return whether a Terminus request visibly ends at a shell prompt."""
    texts = _message_texts(row.get("sample", {}).get("request", {}))
    if not texts:
        return False
    tail = texts[-1].rstrip()
    return bool(re.search(r"(?:^|\n)[^\n]{0,180}(?:[$#])\s*$", tail))


def _bash_categories(command: Any) -> tuple[list[str], bool, list[str]]:
    if not isinstance(command, str) or not command.strip():
        return ["opaque"], True, ["missing_shell_command"]
    if "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in command:
        return ["stop"], False, []
    try:
        analysis = analyze_bash_command(command)
    except (TypeError, ValueError):
        return ["opaque"], True, ["shell_parse_exception"]
    categories = [op.get("category") for op in analysis.get("operations", [])]
    categories = [x for x in categories if x in _ACTION_PRIORITY]
    if not categories:
        categories = [analysis.get("semantic_action") or "opaque"]
    reasons = list(analysis.get("unknown_reasons", []))
    uncertain = bool(reasons) or analysis.get("parse_status") == "error"
    return categories, uncertain, reasons


def _call_projection(call: dict[str, Any], *, harness: str,
                     row: dict[str, Any]) -> tuple[list[str], bool, list[str], str | None]:
    name = call.get("name", "")
    operation = call.get("operation", "")
    args = call.get("arguments") or {}
    if name in _CONTROL_NAMES or operation == "completion_flag":
        value = args.get("task_complete")
        return [], False, [], "complete" if value is True else "continue"
    if operation in _INSPECT_OPERATIONS:
        return ["inspect"], False, [], None
    if operation in _MODIFY_OPERATIONS:
        return ["modify"], False, [], None
    if name in _PLAN_TOOLS or operation == "plan":
        return ["plan"], False, [], None
    if name in _DELEGATE_TOOLS or operation == "delegate":
        return ["delegate"], False, [], None
    if operation == "terminal_wait":
        return ["wait"], False, [], None
    if name in _SHELL_TOOLS or operation == "execute":
        categories, uncertain, reasons = _bash_categories(args.get("command"))
        return categories, uncertain, reasons, None
    if operation == "terminal_input":
        keys = args.get("keystrokes")
        if not isinstance(keys, str) or not keys.strip():
            return ["wait"], False, [], None
        if not _shell_context(row):
            return ["opaque"], True, ["terminal_context_not_verified"], None
        categories, uncertain, reasons = _bash_categories(keys)
        return categories, uncertain, reasons, None
    if operation in {"context_update", "summary"}:
        return ["context_update"], False, [], None
    return ["opaque"], True, [f"unmapped_operation:{operation or name}"], None


def classify_row(row: dict[str, Any]) -> dict[str, Any]:
    """Classify one cached probe row into the common comparison space."""
    decision = row.get("decision", {})
    if not decision.get("countable"):
        return {
            "primary": "invalid", "sequence": ["invalid"], "control": "invalid",
            "action_key": "invalid", "uncertain": True,
            "uncertainty_reasons": ["uncountable_native_response"],
        }
    controls = set(decision.get("control_projection") or [])
    calls = decision.get("calls") or []
    sequence: list[str] = []
    reasons: list[str] = []
    uncertain = False
    completion_control: str | None = None
    for call in calls:
        categories, call_uncertain, call_reasons, control = _call_projection(
            call, harness=decision.get("harness", ""), row=row
        )
        sequence.extend(categories)
        uncertain |= call_uncertain
        reasons.extend(call_reasons)
        if control is not None:
            completion_control = control

    if not calls:
        if "context_update" in controls:
            primary, control = "context_update", "context_update"
        elif "return_to_parent" in controls:
            primary, control = "return", "return"
        elif "root_text_response" in controls:
            primary, control = "stop", "stop"
        else:
            primary, control = "invalid", "invalid"
            uncertain = True
            reasons.append("no_action_no_terminal_control")
        sequence = [primary]
    else:
        if not sequence and completion_control == "complete":
            sequence = ["stop"]
        elif not sequence and completion_control == "continue":
            sequence = ["wait"]
        if not sequence:
            sequence = ["opaque"]
            uncertain = True
            reasons.append("only_controller_calls")
        primary = _dominant(sequence)
        control = completion_control or "none"
        if primary == "stop" and control == "none":
            control = "stop"

    # The primary category is the common action space.  The control suffix is
    # retained so a later estimator can include completion/continuation without
    # silently treating it as an ordinary shell action.
    action_key = f"{primary}|{control}"
    return {
        "primary": primary,
        "sequence": sequence,
        "control": control,
        "action_key": action_key,
        "uncertain": uncertain,
        "uncertainty_reasons": sorted(set(reasons)),
    }


def _smoothed_kl(q: Counter[str], p: Counter[str], actions: tuple[str, ...], alpha: float) -> float:
    q_total = sum(q.values()) + alpha * len(actions)
    p_total = sum(p.values()) + alpha * len(actions)
    return sum(
        ((q[a] + alpha) / q_total)
        * math.log(((q[a] + alpha) / q_total) / ((p[a] + alpha) / p_total))
        for a in actions
    )


def _iter_rows(probe_root: Path, repair_root: Path | None,
               benchmark: str | None = None) -> Iterable[tuple[Path, dict[str, Any]]]:
    benchmark_pattern = benchmark or "*"
    pattern = f"{benchmark_pattern}/*/native/repeat-*/*/call-*/{{role}}-*.json"
    replacements: dict[Path, Path] = {}
    if repair_root is not None:
        for path in repair_root.rglob("*.json"):
            replacements[probe_root / path.relative_to(repair_root)] = path
    for role in ("teacher", "target"):
        for path in sorted(probe_root.glob(pattern.format(role=role))):
            source = replacements.get(path, path)
            yield path, json.loads(source.read_text())


def reclassify(probe_root: Path, *, alpha: float = 0.5,
               repair_root: Path | None = None,
               benchmark: str | None = None) -> dict[str, Any]:
    output_rows: list[dict[str, Any]] = []
    state_rows: dict[tuple[str, str, str, str, str], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: {"teacher": [], "target": []}
    )
    benchmarks_seen: set[str] = set()
    for path, row in _iter_rows(probe_root, repair_root, benchmark):
        projected = classify_row(row)
        relative = path.relative_to(probe_root)
        key = relative.parts[:]
        benchmark_name = key[0]
        benchmarks_seen.add(benchmark_name)
        harness, repeat, task, call = key[1], key[3], key[4], key[5]
        state_key = (benchmark_name, harness, repeat, task, call)
        role = row.get("role")
        if role not in ("teacher", "target"):
            continue
        old = row.get("decision", {})
        output_rows.append({
            "source": str(relative), "benchmark": benchmark_name, "role": role,
            "partition": row.get("partition"),
            "seed": row.get("seed"), "harness": harness, "repeat": repeat,
            "task": task, "call": call, "sample_id": old.get("sample_id"),
            "old_status": old.get("status"), "old_intent": old.get("intent_projection"),
            "old_control": old.get("control_projection"), "projection": projected,
        })
        state_rows[state_key][role].append(projected)

    states: list[dict[str, Any]] = []
    for (benchmark_name, harness, repeat, task, call), roles in sorted(state_rows.items()):
        if not roles["teacher"] or not roles["target"]:
            continue
        q = Counter(x["primary"] for x in roles["teacher"])
        p = Counter(x["primary"] for x in roles["target"])
        jq = Counter(x["action_key"] for x in roles["teacher"])
        jp = Counter(x["action_key"] for x in roles["target"])
        old_q = Counter()
        old_p = Counter()
        for row in output_rows:
            if (row["benchmark"] == benchmark_name and row["harness"] == harness
                    and row["repeat"] == repeat and row["task"] == task
                    and row["call"] == call):
                value = row["old_intent"]
                # The old projection is stored in the scorer, not the row.  Keep
                # this field for audit; the old KL is recomputed separately by
                # the existing scorer's result when needed.
                _ = value
        states.append({
            "benchmark": benchmark_name, "harness": harness, "repeat": repeat,
            "task": task, "call": call,
            "n_teacher": len(roles["teacher"]), "n_target": len(roles["target"]),
            "teacher_primary_counts": dict(q), "target_primary_counts": dict(p),
            "teacher_action_key_counts": dict(jq), "target_action_key_counts": dict(jp),
            "primary_kl": _smoothed_kl(q, p, COMMON_ACTIONS, alpha),
            "joint_kl_observed_union": _smoothed_kl(
                jq, jp, tuple(sorted(set(jq) | set(jp))), alpha
            ),
            "uncertain_teacher": sum(x["uncertain"] for x in roles["teacher"]),
            "uncertain_target": sum(x["uncertain"] for x in roles["target"]),
        })

    by_harness: dict[str, dict[str, Any]] = {}
    for harness in sorted({x["harness"] for x in states}):
        hs = [x for x in states if x["harness"] == harness]
        trajectories: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for state in hs:
            trajectories[(state["repeat"], state["task"])].append(state)
        by_harness[harness] = {
            "states": len(hs),
            "primary_kl_sum": sum(x["primary_kl"] for x in hs),
            "joint_kl_sum": sum(x["joint_kl_observed_union"] for x in hs),
            "uncertain_samples": sum(x["uncertain_teacher"] + x["uncertain_target"] for x in hs),
            "primary_kl_by_trajectory": {
                f"{repeat}/{task}": sum(x["primary_kl"] for x in values)
                for (repeat, task), values in sorted(trajectories.items())
            },
        }
    selection_path = probe_root / "state-selection.json"
    state_selection = json.loads(selection_path.read_text()) if selection_path.exists() else None
    return {
        "schema": "native-common-action-reclassification-v1",
        "projection_version": PROJECTION_VERSION,
        "benchmark": next(iter(benchmarks_seen)) if len(benchmarks_seen) == 1 else None,
        "common_actions": list(COMMON_ACTIONS),
        "alpha": alpha,
        "probe_root": str(probe_root.resolve()),
        "state_selection": state_selection,
        "repair_root": str(repair_root.resolve()) if repair_root else None,
        "method": "Per-harness native decoding followed by a shared finite action projection; complete native invocations retained.",
        "limitation": "Development-set offline relabeling. Categories are lossy and do not establish execution equivalence or teacher optimality.",
        "rows": output_rows,
        "states": states,
        "harnesses": by_harness,
        "fingerprint": _fingerprint({"version": PROJECTION_VERSION, "actions": COMMON_ACTIONS}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repair-root", type=Path)
    parser.add_argument("--benchmark", choices=("tb21", "tb_lite"))
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()
    result = reclassify(args.probe_root, alpha=args.alpha, repair_root=args.repair_root,
                        benchmark=args.benchmark)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
