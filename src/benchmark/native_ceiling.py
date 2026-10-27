"""Score cached native teacher trajectories against a frozen target reference.

This reports finite observed-action projection diagnostics. The complete native
KL-regularized ceiling remains unidentified when action support is open-ended.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from src.adapter.terminal_actions import TERMINAL_ACTIONS, _dominant_semantic_action
from src.benchmark.native_gateway import write_json
from src.hrp.native_measurement import fingerprint

LAYERS = {
    "complete_decision": "local_action_id",
    "intent": "intent_projection",
    "control": "control_projection",
}
PROJECTION_MENU = (*TERMINAL_ACTIONS, "invalid")


def observed_kl(teacher: Counter, reference: Counter, alpha: float) -> float | None:
    """KL(q||p) on the observed union, with the recorded additive smoothing."""
    if not teacher or not reference:
        return None
    actions = teacher.keys() | reference.keys()
    q_total = sum(teacher.values()) + alpha * len(actions)
    p_total = sum(reference.values()) + alpha * len(actions)
    return sum(
        ((teacher[action] + alpha) / q_total)
        * math.log(((teacher[action] + alpha) / q_total)
                   / ((reference[action] + alpha) / p_total))
        for action in actions
    )


def _identity(decision: dict, key: str) -> str | None:
    if not decision.get("countable") or not decision.get("local_action_id"):
        return None
    value = decision.get(key)
    if value is None:
        return None
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return str(value)


def _dominant_v1_outcome(row: dict) -> str | None:
    sample = row.get("sample", {})
    if sample.get("http_status") != 200 or sample.get("error") is not None:
        return None
    decision = row.get("decision", {})
    if not decision.get("countable"):
        return "invalid"
    calls = decision.get("calls") or []
    if not calls:
        return ("stop" if any(control in ("root_text_response", "return_to_parent")
                          for control in decision.get("control_projection") or []) else "opaque")
    categories = []
    for call in calls:
        if call.get("operation") == "completion_flag" and (call.get("arguments") or {}).get("task_complete"):
            categories.append("stop")
        elif call.get("bash_projection"):
            categories.append(call["bash_projection"].get("legacy_semantic_action", "opaque"))
        else:
            categories.extend(call.get("intent_sequence") or ["opaque"])
    categories = [category for category in categories if category in TERMINAL_ACTIONS]
    if categories and all(category == "stop" for category in categories):
        return "stop"
    return _dominant_semantic_action([category for category in categories if category != "stop"])


def _projected_kl(rows: dict[str, list[dict]], expected: dict[str, int], alpha: float) -> dict | None:
    if any(len(rows[role]) != expected[role] for role in ("teacher", "target")):
        return None
    outcomes = {role: [_dominant_v1_outcome(row) for row in values] for role, values in rows.items()}
    if any(None in values for values in outcomes.values()):
        return None
    counts = {role: Counter(values) for role, values in outcomes.items()}
    q_total = expected["teacher"] + alpha * len(PROJECTION_MENU)
    p_total = expected["target"] + alpha * len(PROJECTION_MENU)
    kl = sum(
        ((counts["teacher"][action] + alpha) / q_total)
        * math.log(((counts["teacher"][action] + alpha) / q_total)
                   / ((counts["target"][action] + alpha) / p_total))
        for action in PROJECTION_MENU
    )
    return {"kl_teacher_reference": kl,
            "teacher_invalid_samples": counts["teacher"]["invalid"],
            "reference_invalid_samples": counts["target"]["invalid"]}


def score_state(path: Path, *, expected: dict[str, int], alpha: float,
                overrides: dict[Path, dict] | None = None) -> dict:
    validation = path / "replay_validation.json"
    if not validation.exists():
        return {"status": "probe_unavailable", "layers": {}}
    check = json.loads(validation.read_text())
    if not check.get("source_unchanged") or not check.get("same_context_and_tools"):
        return {"status": "frozen_request_validation_failed", "layers": {}}
    overrides = overrides or {}
    rows = {
        role: [overrides[p] if p in overrides else json.loads(p.read_text())
               for p in sorted(path.glob(f"{role}-*.json"))]
        for role in ("teacher", "target")
    }
    sample_counts = {role: len(values) for role, values in rows.items()}
    projected = _projected_kl(rows, expected, alpha)
    valid = all(sample_counts[role] == expected[role] for role in rows)
    failures = {}
    for role, values in rows.items():
        failures[role] = sum(
            row.get("sample", {}).get("http_status") != 200
            or row.get("sample", {}).get("error") is not None
            or not row.get("decision", {}).get("countable")
            for row in values
        )
        valid &= failures[role] == 0
    layers = {}
    if valid:
        for layer, key in LAYERS.items():
            counts = {
                role: Counter(_identity(row["decision"], key) for row in values)
                for role, values in rows.items()
            }
            if any(None in counts[role] for role in rows):
                valid = False
                break
            value = observed_kl(counts["teacher"], counts["target"], alpha)
            layers[layer] = {
                "kl_teacher_reference_observed_union": value,
                "observed_action_count": len(counts["teacher"].keys() | counts["target"].keys()),
                "teacher_distinct": len(counts["teacher"]),
                "reference_distinct": len(counts["target"]),
            }
    return {
        "status": "observed_union_estimated" if valid else "incomplete_probability_samples",
        "sample_counts": sample_counts,
        "failed_or_unresolved_samples": failures,
        "cross_model_private_replay_unverified": any(
            row.get("sample", {}).get("cross_model_private_replay_unverified", False)
            for values in rows.values() for row in values
        ),
        "dominant_v1_invalid_projection": projected,
        "repaired_samples": sum(p in overrides for role in rows for p in path.glob(f"{role}-*.json")),
        "layers": layers if valid else {},
    }


def score(args: argparse.Namespace) -> dict:
    teacher_root = args.teacher_experiment.resolve()
    probe_root = args.probe_experiment.resolve()
    teacher_manifest = json.loads((teacher_root / "manifest.json").read_text())
    probe_manifest = json.loads((probe_root / "manifest.json").read_text())
    if probe_manifest.get("source_trajectory_fingerprint") != teacher_manifest["fingerprint"]:
        raise ValueError("Probe experiment does not use the selected teacher trajectories")
    if probe_manifest.get("validation_only") or probe_manifest.get("mode") != "frozen_native_replay":
        raise ValueError("Ceiling scoring requires complete frozen native replay")
    if teacher_manifest.get("rollout_model_role") != "teacher":
        raise ValueError("Ceiling occupancy must come from teacher trajectories")
    if teacher_manifest["target"] != probe_manifest["target"] or teacher_manifest["teacher"] != probe_manifest["teacher"]:
        raise ValueError("Target reference or teacher changed between trajectory and probe runs")
    benchmarks = list(teacher_manifest.get("benchmarks", []))
    benchmark = getattr(args, "benchmark", None)
    if benchmark is None:
        if len(benchmarks) != 1:
            raise ValueError("Specify a benchmark when the manifest contains multiple benchmarks")
        benchmark = benchmarks[0]
    if benchmark not in benchmarks or benchmark not in teacher_manifest.get("tasks", {}):
        raise ValueError(f"Unknown benchmark {benchmark}; available: {benchmarks}")
    if teacher_manifest["interfaces"] != ["native"]:
        raise ValueError("This scorer expects native trajectories")
    planned = list(teacher_manifest["tasks"][benchmark])
    selected_tasks = getattr(args, "task_ids", None) or planned
    unknown = sorted(set(selected_tasks) - set(planned))
    if unknown:
        raise ValueError(f"Unknown {benchmark} task IDs: {unknown}")
    expected = {
        "target": probe_manifest["hrp"]["n_target"],
        "teacher": probe_manifest["hrp"]["n_teacher"],
    }
    alpha = probe_manifest["hrp"]["alpha"]
    tau = probe_manifest["hrp"]["tau"]
    if alpha <= 0 or tau <= 0:
        raise ValueError("Positive alpha and tau are required")
    overrides = {}
    repair_audit = []
    repair_root = getattr(args, "repair_root", None)
    if repair_root is not None:
        repair_root = repair_root.resolve()
        for replacement in sorted(repair_root.rglob("*.json")):
            original_path = probe_root / replacement.relative_to(repair_root)
            original = json.loads(original_path.read_text())
            repaired = json.loads(replacement.read_text())
            for key in ("role", "partition", "seed", "measurement_version",
                        "source_state_fingerprint", "source_request_fingerprint", "config_fingerprint"):
                if original[key] != repaired[key]:
                    raise ValueError(f"Repair changes sample identity {key}: {replacement}")
            if not repaired.get("repair"):
                raise ValueError(f"Repair provenance missing: {replacement}")
            overrides[original_path] = repaired
            repair_audit.append({"sample": str(original_path.relative_to(probe_root)),
                                 "original_sha256": hashlib.sha256(original_path.read_bytes()).hexdigest(),
                                 "replacement_sha256": hashlib.sha256(replacement.read_bytes()).hexdigest()})
    output = {
        "method": "teacher occupancy; official terminal reward minus tau times sum of KL(q||p)",
        "teacher_experiment": str(teacher_root),
        "probe_experiment": str(probe_root),
        "benchmark": benchmark,
        "target_reference": teacher_manifest["target"]["id"],
        "teacher": teacher_manifest["teacher"]["id"],
        "gamma": 1,
        "tau": tau,
        "action_scope": "native decisions saved; plug-in KL only on observed action unions",
        "full_native_ceiling_proxy": None,
        "full_native_ceiling_reason": "Open native action support and execution equivalence are unverified",
        "teacher_proxy_projection": "dominant-v1 seven classes plus an invalid native-response outcome",
        "teacher_proxy_limitation": "Projected KL omits within-class action and argument differences; it is not a certified native-control lower bound",
        "repair_overrides": repair_audit,
        "harnesses": {},
    }
    for harness in teacher_manifest["harnesses"]:
        trajectories = []
        for repeat in range(1, teacher_manifest["repeats"] + 1):
            for task in selected_tasks:
                relative = Path(benchmark) / harness / "native" / f"repeat-{repeat}" / task
                run = teacher_root / relative
                result_path = run / "result.json"
                if not result_path.exists():
                    trajectories.append({"task_id": task, "repeat": repeat, "status": "trajectory_unavailable"})
                    continue
                result = json.loads(result_path.read_text())
                reward = result.get("reward")
                if reward not in (0, 0.0, 1, 1.0):
                    raise ValueError(f"Expected official binary reward: {result_path}")
                wires = {}
                for path in sorted((run / "wire").glob("*/record.json")):
                    record = json.loads(path.read_text())
                    state_id = record.get("state_id")
                    if not state_id or "messages" not in record.get("request", {}):
                        continue
                    if state_id not in wires or record.get("status") == "completed":
                        wires[state_id] = (path, record)
                penalties = {layer: 0.0 for layer in LAYERS}
                projected_kl_sum = 0.0
                projected_complete = True
                states = []
                for state_id, (path, record) in wires.items():
                    probed = probe_root / relative / record.get("attempt", path.parent.name)
                    diagnostic = score_state(probed, expected=expected, alpha=alpha, overrides=overrides)
                    diagnostic = {"state_id": state_id, "attempt": record.get("attempt"), **diagnostic}
                    states.append(diagnostic)
                    projection = diagnostic.get("dominant_v1_invalid_projection")
                    if projection is None:
                        projected_complete = False
                    else:
                        projected_kl_sum += projection["kl_teacher_reference"]
                    if diagnostic["status"] == "observed_union_estimated":
                        for layer in LAYERS:
                            penalties[layer] += diagnostic["layers"][layer]["kl_teacher_reference_observed_union"]
                covered = sum(row["status"] == "observed_union_estimated" for row in states)
                complete = bool(result.get("generation", {}).get("status")
                                in {"completed", "native_loop_returned", "Submitted"}
                                and covered == len(wires) and len(wires) > 0)
                projected_complete = bool(projected_complete and len(wires) > 0
                    and result.get("generation", {}).get("status")
                    in {"completed", "native_loop_returned", "Submitted"})
                trajectories.append({
                    "task_id": task, "repeat": repeat, "status": "complete_observed_union" if complete else "incomplete",
                    "official_reward": reward, "recorded_decision_states": len(wires),
                    "estimated_decision_states": covered,
                    "private_replay_unverified_states": sum(
                        row.get("cross_model_private_replay_unverified", False) for row in states
                    ),
                    "projected_kl_sum": penalties if complete else None,
                    "projected_teacher_return": {
                        layer: reward - tau * value for layer, value in penalties.items()
                    } if complete else None,
                    "dominant_v1_invalid_kl_sum": projected_kl_sum if projected_complete else None,
                    "dominant_v1_invalid_teacher_proxy": (
                        reward - tau * projected_kl_sum if projected_complete else None),
                    "states": states,
                })
        complete = all(row["status"] == "complete_observed_union" for row in trajectories)
        projected_complete = all(row["dominant_v1_invalid_teacher_proxy"] is not None for row in trajectories)
        output["harnesses"][harness] = {
            "planned_trajectories": len(trajectories),
            "completed_projected_trajectories": sum(row["status"] == "complete_observed_union" for row in trajectories),
            "official_teacher_success": (
                sum(row["official_reward"] for row in trajectories) / len(trajectories)
                if all("official_reward" in row for row in trajectories) else None
            ),
            "observed_union_projected_teacher_return": {
                layer: sum(row["projected_teacher_return"][layer] for row in trajectories) / len(trajectories)
                for layer in LAYERS
            } if complete else None,
            "dominant_v1_invalid_teacher_proxy": (
                sum(row["dominant_v1_invalid_teacher_proxy"] for row in trajectories) / len(trajectories)
                if projected_complete else None),
            "dominant_v1_invalid_reference_penalty": (
                tau * sum(row["dominant_v1_invalid_kl_sum"] for row in trajectories) / len(trajectories)
                if projected_complete else None),
            "dominant_v1_invalid_complete_trajectories": sum(
                row["dominant_v1_invalid_teacher_proxy"] is not None for row in trajectories),
            "trajectories": trajectories,
        }
    output["fingerprint"] = fingerprint({
        "teacher": teacher_manifest["fingerprint"],
        "probes": probe_manifest["fingerprint"],
        "tau": tau, "alpha": alpha, "method": output["method"],
        "teacher_proxy_projection": output["teacher_proxy_projection"],
        "repair_overrides": repair_audit,
    })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-experiment", type=Path, required=True)
    parser.add_argument("--probe-experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", choices=("tb21", "tb_lite"))
    parser.add_argument("--task-ids", nargs="+")
    parser.add_argument("--repair-root", type=Path,
                        help="Overlay audited sample repairs by probe-relative path")
    args = parser.parse_args()
    write_json(args.output, score(args))


if __name__ == "__main__":
    main()
