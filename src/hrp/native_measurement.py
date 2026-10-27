"""Conservative, state-local diagnostics for open-ended native decisions.

These measurements are teacher-relative diagnostics, not the finite-action
full-space HRP energy and not a calibrated prediction of benchmark success.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import Counter
from typing import Any

from jsonschema import Draft202012Validator

MEASUREMENT_VERSION = "native-layered-pairs-v2"


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def decode_response(body: str, *, streaming: bool = False) -> dict:
    """Retain raw transport separately; assemble a complete ordered decision."""
    if not streaming:
        return json.loads(body)
    message: dict = {"role": "assistant", "content": ""}
    calls: dict[int, dict] = {}
    result: dict = {"choices": [{"message": message, "finish_reason": None}]}
    for line in body.splitlines():
        if not line.startswith("data: ") or line[6:].strip() == "[DONE]":
            continue
        item = json.loads(line[6:])
        if item.get("usage"):
            result["usage"] = item["usage"]
        if item.get("error"):
            result["error"] = item["error"]
        for choice in item.get("choices", []):
            if choice.get("index", 0) != 0:
                continue
            delta = choice.get("delta", {})
            for key in ("content", "reasoning_content", "reasoning"):
                if isinstance(delta.get(key), str):
                    message[key] = message.get(key, "") + delta[key]
            for call in delta.get("tool_calls", []):
                merged = calls.setdefault(call.get("index", 0),
                                          {"type": "function", "function": {"name": "", "arguments": ""}})
                if call.get("id"):
                    merged["id"] = call["id"]
                for key in ("name", "arguments"):
                    merged["function"][key] += call.get("function", {}).get(key, "")
            if choice.get("finish_reason"):
                result["choices"][0]["finish_reason"] = choice["finish_reason"]
    if calls:
        message["tool_calls"] = [calls[key] for key in sorted(calls)]
    return result


# Tool names have different range/truncation/locking contracts across frameworks.
# This table identifies operations, never establishes cross-tool equivalence.
_OPERATIONS = {
    "read": ("inspect", "read"), "read_file": ("inspect", "read"),
    "ls": ("inspect", "list"), "glob": ("inspect", "search"),
    "find": ("inspect", "search"), "grep": ("inspect", "search"),
    "write": ("modify", "write"), "write_file": ("modify", "write"),
    "edit": ("modify", "edit"), "edit_file": ("modify", "edit"),
    "bash": ("opaque", "execute"), "execute": ("opaque", "execute"),
    "task": ("run", "delegate"), "subagent": ("run", "delegate"),
    "write_todos": ("run", "plan"), "todo_write": ("run", "plan"), "delete": ("modify", "delete"),
    "terminal_keystrokes": ("run", "terminal_input"),
    "terminal_completion_flag": ("opaque", "completion_flag"),
}


def decision_record(response: dict, *, state_id: str, harness: str,
                    sample_id: str, environment_version: str | None,
                    tool_definitions: list | None = None, call_kind: str = "decision",
                    boundary: dict | None = None) -> dict:
    """Normalize JSON representation only, retain every argument and call order.

    Exact invocation identities are deliberately distinct from verified semantic
    classes. Counting a script/terminal/delegation invocation does not prove its
    execution effects or merge it with a differently expressed operation.
    """
    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    finish = choice.get("finish_reason")
    calls = message.get("tool_calls") or []
    content = message.get("content")
    boundary = boundary or {}
    call_kind = response.get("native_call_kind", call_kind)
    is_summary = any(part in call_kind.lower() for part in ("summary", "summar", "compaction"))
    is_child = boundary.get("agent_role") == "subagent" or "subagent" in call_kind
    visible = bool(content.strip()) if isinstance(content, str) else bool(content)
    control = "context_update" if is_summary else "tool_execution" if calls else (
        "return_to_parent" if is_child else "root_text_response")
    status = "tool_calls" if calls else "text_response" if visible else "empty"
    if response.get("native_parser_error") and (calls or visible):
        status = "illegal_call"
    if finish in ("length", "max_tokens"):
        status = "length_truncated"
    if response.get("error"):
        status = "provider_error"
    records, identities = [], []
    schemas = {tool.get("function", {}).get("name"): tool.get("function", {}).get("parameters", {})
               for tool in tool_definitions or []}
    for order, call in enumerate(calls):
        function = call.get("function", {})
        name = function.get("name", "")
        raw_args = function.get("arguments", {})
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            if not isinstance(args, dict):
                raise ValueError("tool arguments must be an object")
        except (ValueError, TypeError):
            args = None
            status = "illegal_call" if status not in ("length_truncated", "provider_error") else status
        validation_errors = []
        if args is not None and name in schemas:
            validation_errors = [error.message for error in Draft202012Validator(schemas[name]).iter_errors(args)]
            if validation_errors and status not in ("length_truncated", "provider_error"):
                status = "illegal_call"
        elif tool_definitions is not None and name not in schemas:
            validation_errors = ["Tool not present in the frozen native tool definitions"]
            if status not in ("length_truncated", "provider_error"):
                status = "illegal_call"
        intent, operation = _OPERATIONS.get(name, ("opaque", "unknown"))
        if operation == "terminal_input" and args is not None and args.get("keystrokes") == "":
            operation = "terminal_wait"
        bash_projection = call.get("bash_projection")
        intent_sequence = ([op["category"] for op in bash_projection["operations"] if op.get("category")]
                           if bash_projection else [intent])
        if not intent_sequence:
            intent_sequence = ["opaque"]
        objects = {key: args[key] for key in ("path", "file_path", "file", "cwd", "pattern",
                    "query", "command", "offset", "limit", "start_line", "end_line",
                    "keystrokes", "duration", "task_complete", "subagent_type", "agent_type",
                    "description", "prompt", "task")
                   if args is not None and key in args}
        identity = {"tool": name, "arguments": args, "order": order}
        identities.append(identity)
        records.append({"order": order, "raw_call": call, "name": name,
                        "arguments": args, "intent": intent, "operation": operation,
                        "intent_sequence": intent_sequence, "bash_projection": bash_projection,
                        "objects": objects, "state_id": state_id,
                        "environment_version": environment_version,
                        "validation_errors": validation_errors,
                        "equivalence": {"basis": "JSON object-key order only; full arguments retained",
                            "cross_tool_merge": False, "execution_verified": False}})
    controls = [control]
    for record in records:
        if record["operation"] == "delegate":
            controls.append("delegate")
        if record["operation"] in ("terminal_input", "terminal_wait"):
            controls.append(record["operation"])
        if record["operation"] == "completion_flag" and record["arguments"] is not None:
            controls.append(("completion_confirmation" if boundary.get("pending_completion") else
                "completion_request") if record["arguments"].get("task_complete") else "continue")
    scope = {"state": state_id, "harness": harness,
                               "environment_version": environment_version,
             "tools": tool_definitions, "call_kind": call_kind, "boundary": boundary}
    request_key = fingerprint({**scope, "calls": identities, "control": controls,
                               "status": status, "content": content})
    parsed = status in ("tool_calls", "text_response") and not response.get("native_parser_error") and all(
        r["name"] and r["arguments"] is not None and not r["validation_errors"] for r in records)
    invocation_key = fingerprint({**scope, "calls": identities, "control": controls,
        # A summary/return consists of text; it must not collapse to one action.
        "text": None if records else content}) if parsed else None
    return {"measurement_version": MEASUREMENT_VERSION,
            "sample_id": sample_id, "state_id": state_id, "harness": harness,
            "raw_response": response, "finish_reason": finish, "status": status,
            "flags": {"empty_public_reply": not calls and not visible,
                      "invalid_call": bool(response.get("native_parser_error")) or any(r["arguments"] is None or r["validation_errors"] for r in records),
                      "length_truncated": finish in ("length", "max_tokens"),
                      "provider_error": bool(response.get("error"))},
            "calls": records, "request_identity": request_key,
            "raw_decision_id": fingerprint(choice), "local_action_id": request_key if parsed else None,
            "invocation_id": invocation_key, "countable": parsed,
            "call_kind": call_kind, "boundary": boundary, "control_projection": controls,
            "invocation_projection_is_lossy": bool(records and visible),
            "execution_semantics": "unverified", "environment_available": bool(environment_version),
            "verified_action_id": None,
            "decoder_scope": ("native decoder failed; raw transport retained" if response.get("native_decoder_failed") else
                "native provider decoder; tools not executed" if response.get("native_decoder") else "raw provider response; native decoder not verified"),
            "intent_projection": [r["intent_sequence"] for r in records] or [[
                "stop" if parsed and control == "root_text_response" else "opaque"]],
            "object_projection": [{"operation": r["operation"], "objects": r["objects"],
                                   "shell_operations": (r["bash_projection"] or {}).get("operations")} for r in records],
            "projections_are_lossy": True,
            "unresolved_id": None if parsed else f"unresolved:{sample_id}",
            "usage": response.get("usage", {})}


def admit_equivalence(left: dict, right: dict, evidence: dict) -> str | None:
    """Accept only explicit same-state, contract-backed equivalence evidence.

    A matching file diff or one matching execution alone is insufficient.
    Execution evidence must include native feedback and side-effect accounting.
    """
    if left["state_id"] != right["state_id"] or evidence.get("state_id") != left["state_id"]:
        return None
    required = ("same_initial_snapshot", "file_changes_equal", "visible_feedback_equal",
                "exit_status_equal", "side_effects_accounted", "deterministic_contract")
    if not all(evidence.get(key) is True for key in required) or not evidence.get("contract_reference"):
        return None
    if set(evidence.get("invocations", [])) != {left["request_identity"], right["request_identity"]}:
        return None
    return "verified:" + fingerprint({"state": left["state_id"], "evidence": evidence})


def wilson(count: int, total: int) -> list[float] | None:
    if not total:
        return None
    z = 1.959963984540054
    p, d = count / total, 1 + z*z / total
    center = (p + z*z / (2*total)) / d
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / d
    return [max(0., center-half), min(1., center+half)]


def _layer_diagnostics(samples: list[dict], key: str, *, tau: float,
                       alpha: float, min_pair_count: int) -> dict:
    """The same discovery/estimation rule for every declared representation."""
    def identity(sample):
        decision = sample["decision"]
        if not decision.get("local_action_id"):
            return None
        value = decision.get(key)
        return json.dumps(value, sort_keys=True, ensure_ascii=False) if isinstance(value, (list, dict)) else value

    discovery = [s for s in samples if s["partition"] == "discovery"]
    candidates = sorted({identity(s) for s in discovery if identity(s) is not None})
    counts: dict[str, Counter] = {}
    roles = {}
    for role in ("target", "teacher"):
        discovered = [s for s in discovery if s["role"] == role]
        rows = [s for s in samples if s["role"] == role and s["partition"] == "estimation"]
        c = Counter(identity(r) for r in rows if identity(r) is not None)
        counts[role] = c
        n = len(rows)
        covered = sum(c[k] for k in candidates)
        roles[role] = {"n": n, "all_local_action_counts": dict(c),
                       "discovery": {"n": len(discovered),
                           "local_action_counts": dict(Counter(identity(r) for r in discovered if identity(r) is not None)),
                           "unresolved_count": sum(identity(r) is None for r in discovered),
                           "status_counts": dict(Counter(r["decision"]["status"] for r in discovered))},
                       "candidate_counts": {k: c[k] for k in candidates},
                       "candidate_covered_count": covered,
                       "candidate_coverage": covered/n if n else None,
                       "candidate_coverage_wilson95": wilson(covered, n),
                       "unresolved_count": sum(identity(r) is None for r in rows),
                       "unresolved_mass": sum(identity(r) is None for r in rows)/n if n else None,
                       "outside_candidate_count": sum(value for action,value in c.items() if action not in candidates),
                       "zero_count_candidates": sum(c[k] == 0 for k in candidates),
                       "status_counts": dict(Counter(r["decision"]["status"] for r in rows))}
    pairs = []
    for a, b in itertools.combinations(candidates, 2):
        values = [counts[role][key] for role in ("target", "teacher") for key in (a, b)]
        delta = tau * (math.log((values[0]+alpha)/(values[1]+alpha)) -
                       math.log((values[2]+alpha)/(values[3]+alpha)))
        supported = min(values) >= min_pair_count
        se = tau * math.sqrt(sum(1/(v+alpha) for v in values))
        pairs.append({"a": a, "b": b, "counts_target_a_b_teacher_a_b": values,
                      "supported": supported, "smoothed_log_odds_difference": delta,
                      "approximate_log_odds_95": [delta-1.96*se, delta+1.96*se],
                      "squared_difference": delta*delta if supported else None})
    supported_pairs = [p for p in pairs if p["supported"]]
    return {"candidate_source": "independent discovery partition; free generation throughout",
            "candidates": candidates, "roles": roles, "pairs": pairs,
            "supported_pair_count": len(supported_pairs), "candidate_pair_count": len(pairs),
            "status": "local_pairs_supported" if supported_pairs else "insufficient_support",
            "supported_pair_subset_mean": sum(p["squared_difference"] for p in supported_pairs)/len(supported_pairs) if supported_pairs else None}


def state_diagnostics(samples: list[dict], *, state_id: str, tau: float = .1,
                      alpha: float = .5, min_pair_count: int = 3) -> dict:
    """Retain full decisions, invocation projections and explicitly lossy views."""
    keys = {"complete_decision": "local_action_id", "invocation": "invocation_id",
            "intent": "intent_projection", "operation_object": "object_projection",
            "control": "control_projection", "verified_semantic": "verified_action_id"}
    layers = {name: {**_layer_diagnostics(samples, key, tau=tau, alpha=alpha, min_pair_count=min_pair_count),
                     "is_projection": name in ("invocation", "intent", "operation_object", "control"),
                     "requires_execution_evidence": name == "verified_semantic",
                     "identity_basis": ("admitted same-state contract and execution evidence" if name == "verified_semantic" else
                                        "normalized recorded representation; no execution equivalence claim")}
              for name, key in keys.items()}
    return {**layers["complete_decision"], "state_id": state_id, "measurement_version": MEASUREMENT_VERSION,
            "layers": layers,
            "local_pair_error": None,
            "verified_semantic_error": None, "full_space_hrp_energy": None,
            "predicted_success": None, "harness_rank": None,
            "limitations": ["Exact invocation pairs are not verified cross-tool semantic pairs.",
                            "Intervals are approximate marginal intervals, not simultaneous guarantees.",
                            "Unresolved samples remain distinct; no outcome calibration has been supplied."]}


def summarize_diagnostics(states: list[dict]) -> dict:
    """Report measured subsets and their missingness, never impute missing errors."""
    layers = {}
    names = sorted({name for state in states for name in state.get("layers", {})})
    for name in names:
        selected = [state["layers"][name] for state in states if name in state.get("layers", {})]
        estimates = [row["supported_pair_subset_mean"] for row in selected if row["supported_pair_subset_mean"] is not None]
        roles = {}
        for role in ("target", "teacher"):
            count = sum(row["roles"][role]["n"] for row in selected)
            covered = sum(row["roles"][role]["candidate_covered_count"] for row in selected)
            roles[role] = {"estimation_samples": count, "candidate_covered_samples": covered,
                "pooled_candidate_coverage": covered/count if count else None,
                "unresolved_samples": sum(row["roles"][role]["unresolved_count"] for row in selected)}
        layers[name] = {"all_boundaries": len(states), "sampled_boundaries": len(selected),
            "supported_boundaries": len(estimates), "unsupported_or_unavailable_boundaries": len(states)-len(estimates),
            "supported_state_subset_mean": sum(estimates)/len(estimates) if estimates else None,
            "scope": "unweighted observed supported subset; not complete occupancy-weighted error", "roles": roles}
    return {"boundaries": len(states), "status_counts": dict(Counter(s["status"] for s in states)),
            "call_kind_counts": dict(Counter(s.get("call_kind", "unclassified") for s in states)),
            "layers": layers, "full_space_energy": None, "predicted_success": None, "harness_ranking": None}
