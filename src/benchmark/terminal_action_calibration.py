"""Offline Terminal intent audit and blinded annotation; never executes commands."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from src.adapter.terminal_actions import (
    CLASSIFIER_FINGERPRINT, CLASSIFIER_VERSION, PARSER_VERSIONS, TERMINAL_ACTION_PROJECTION,
    TERMINAL_ACTIONS, _dominant_semantic_action, analyze_bash_command,
)


def reclassify_action(action: dict[str, Any]) -> dict[str, Any]:
    analyses = [analyze_bash_command(call["command"]) for call in action["calls"]]
    return {
        "v1": _dominant_semantic_action([a["legacy_semantic_action"] for a in analyses]) if analyses else "stop",
        "v2": _dominant_semantic_action([a["semantic_action"] for a in analyses]) if analyses else "stop",
        "analyses": analyses,
        "compound": len(analyses) > 1 or any(len(a["operations"]) > 1 or a["structure"] for a in analyses),
        "uncertain": any(a["unknown_reasons"] for a in analyses),
        "parse_error": any(a["parse_status"] == "error" for a in analyses),
        "termination_reason": action.get("termination_reason") or (
            "rejected_tool_calls" if action.get("rejected_tool_calls") and not analyses
            else "no_tool_calls" if not analyses and action.get("assistant_text")
            else "empty_response" if not analyses else None
        ),
    }


def read_records(root: Path, task_ids: set[str] | None = None) -> Iterable[dict[str, Any]]:
    """Read exact rollout actions and new raw probes, not lossy count pickles."""
    for path in sorted(root.rglob("*terminal-actions.json")):
        config = json.loads((path.parents[1] / "config.json").read_text())
        task_id = "terminal-bench/" + Path(config["task"]["path"]).parent.name
        if task_ids is not None and task_id not in task_ids:
            continue
        harness = {"pi": "pi", "deepagents": "deepagents", "deepseek": "dsh", "terminal": "terminal"}.get(path.name.split("-")[0], path.name.split("-")[0])
        for index, action in enumerate(json.loads(path.read_text())):
            yield {"source": str(path.resolve()), "index": index, "kind": "rollout",
                   "task_id": task_id, "harness": harness, "action": action}
    for path in sorted(root.rglob("samples.jsonl")):
        if path.parent.name != "probes":
            continue
        for index, line in enumerate(path.read_text().splitlines()):
            row = json.loads(line)
            if task_ids is not None and row["task_id"] not in task_ids:
                continue
            if row.get("schema") != "terminal-probe-v2":
                raise ValueError(f"Unsupported raw probe schema: {path}:{index + 1}")
            yield {"source": str(path.resolve()), "index": index, "kind": "probe",
                   "task_id": row["task_id"], "harness": row["harness"], "action": row["action"],
                   "state_id": row["state_id"], "model": row["model"], "role": row["role"],
                   "sample_id": row["sample_id"]}


def audit(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    probe_counts: dict[tuple[str, ...], dict[str, Counter]] = {}
    for row in records:
        result = reclassify_action(row["action"])
        key = f"{row['kind']}/{row['harness']}"
        group = groups.setdefault(key, {"decisions": 0, "v1": Counter(), "v2": Counter(), "migrations": Counter(), "changed": 0, "stored": Counter(), "stored_to_v2": Counter(), "stored_changed": 0, "compound": 0, "uncertain": 0, "parse_error": 0, "termination": Counter()})
        group["decisions"] += 1
        for version in ("v1", "v2"):
            group[version][result[version]] += 1
        group["migrations"][f"{result['v1']}->{result['v2']}"] += 1
        group["changed"] += result["v1"] != result["v2"]
        stored = row["action"]["semantic_action"]
        group["stored"][stored] += 1
        group["stored_to_v2"][f"{stored}->{result['v2']}"] += 1
        group["stored_changed"] += stored != result["v2"]
        for flag in ("compound", "uncertain", "parse_error"):
            group[flag] += bool(result[flag])
        if result["termination_reason"]:
            group["termination"][result["termination_reason"]] += 1
        if row["kind"] == "probe":
            state_key = tuple(row[k] for k in ("harness", "task_id", "state_id", "role", "model"))
            counts = probe_counts.setdefault(state_key, {"v1": Counter(), "v2": Counter()})
            for version in ("v1", "v2"):
                counts[version][result[version]] += 1
    return {
        "projection": TERMINAL_ACTION_PROJECTION, "fingerprint": CLASSIFIER_FINGERPRINT,
        "groups": groups,
        "probe_counts": [dict(zip(("harness", "task_id", "state_id", "role", "model"), key), **counts) for key, counts in sorted(probe_counts.items())],
        "limitation": "Intent migrations are not accuracy, return gains, or behavioral equivalence. Historical count-only probes cannot be relabeled.",
    }


def blinded_action(action: dict[str, Any]) -> dict[str, Any]:
    return {
        "calls": [{"tool_name": call["tool_name"], "arguments": call["arguments"]} for call in action["calls"]],
        "assistant_text": action.get("assistant_text", ""),
        "rejected_tool_calls": action.get("rejected_tool_calls", []),
    }


def build_annotations(records: list[dict[str, Any]], *, size: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if size < 2:
        raise ValueError("Annotation size must be at least 2")
    rng = random.Random(seed)
    task_ids = sorted({row["task_id"] for row in records})
    if len(task_ids) < 2:
        raise ValueError("Need at least two tasks for a disjoint annotation validation split")
    rng.shuffle(task_ids)
    cut = max(1, min(len(task_ids) - 1, round(len(task_ids) * 0.6)))
    calibration = set(task_ids[:cut])
    strata: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for row in records:
        public = blinded_action(row["action"])
        # IDs are serialization artifacts, not part of an executable command.
        identity = json.dumps(public, sort_keys=True, ensure_ascii=False)
        if identity in seen:
            continue
        seen.add(identity)
        result = reclassify_action(row["action"])
        split = "calibration" if row["task_id"] in calibration else "validation"
        stratum = (split, row["harness"], result["v1"], "compound" if result["compound"] else "single")
        strata[stratum].append({**row, "public": public, "prediction": result, "split": split})
    for values in strata.values():
        rng.shuffle(values)
    keys = sorted(strata)
    available = {key: len(values) for key, values in strata.items()}
    rng.shuffle(keys)
    selected: list[tuple[tuple[str, ...], dict[str, Any]]] = []
    while len(selected) < size and any(strata.values()):
        for key in keys:
            if strata[key] and len(selected) < size:
                selected.append((key, strata[key].pop()))
    rng.shuffle(selected)
    annotations, private = [], []
    for key, row in selected:
        sample_id = hashlib.sha256(f"{seed}:{row['source']}:{row['index']}".encode()).hexdigest()[:20]
        annotations.append({"id": sample_id, **row["public"], "label": None, "uncertain": None, "notes": ""})
        private.append({"id": sample_id, "source": row["source"], "index": row["index"], "harness": row["harness"], "task_id": row["task_id"], "split": row["split"], "stratum": list(key), "stratum_available": available[key], "v1": row["prediction"]["v1"], "v2": row["prediction"]["v2"]})
    manifest = {
        "schema": "terminal-annotation-v2", "classifier_version": CLASSIFIER_VERSION,
        "fingerprint": CLASSIFIER_FINGERPRINT, "seed": seed, "requested": size,
        "parser_versions": PARSER_VERSIONS,
        "selected": len(annotations), "source_decisions": len(records),
        "unique_decisions": len(seen), "calibration_tasks": sorted(calibration),
        "validation_tasks": sorted(set(task_ids) - calibration),
        "selection": "Round robin over split/harness/V1 category/compound strata; deduplicated public decisions.",
        "label_status": "pending_human_annotation",
        "scope": "Classifier annotation split inside pilot tasks; not a held-out Section 5.3 benchmark evaluation.",
    }
    return annotations, private, manifest


def score_annotations(annotations: list[dict[str, Any]], private: list[dict[str, Any]]) -> dict[str, Any]:
    keys = {row["id"]: row for row in private}
    if len(keys) != len(private) or len({row["id"] for row in annotations}) != len(annotations):
        raise ValueError("Duplicate annotation IDs")
    if {row["id"] for row in annotations} != set(keys):
        raise ValueError("Annotation and provenance IDs must match")
    groups: dict[str, Any] = {}
    for row in annotations:
        key = keys[row["id"]]
        group = groups.setdefault(key["split"], {"total": 0, "labeled": 0, "uncertain": 0, "v1_correct": 0, "v2_correct": 0, "v1_confusion": Counter(), "v2_confusion": Counter()})
        group["total"] += 1
        label = row.get("label")
        if label is None or label == "":
            continue
        if label not in TERMINAL_ACTIONS:
            raise ValueError(f"Invalid human label for {row['id']}: {label!r}")
        group["labeled"] += 1
        group["uncertain"] += row.get("uncertain") is True
        for version in ("v1", "v2"):
            group[f"{version}_correct"] += key[version] == label
            group[f"{version}_confusion"][f"{label}->{key[version]}"] += 1
    for group in groups.values():
        for version in ("v1", "v2"):
            group[f"{version}_sample_accuracy"] = group[f"{version}_correct"] / group["labeled"] if group["labeled"] else None
    return {"splits": groups, "status": "complete" if all(g["labeled"] == g["total"] for g in groups.values()) and groups else "pending_human_annotation", "limitation": "Stratified sample accuracy, not deployment accuracy or TB2 task return."}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "build"):
        command = commands.add_parser(name)
        command.add_argument("--input", type=Path, required=True)
        command.add_argument("--tasks-from", type=Path, help="Pilot summary with explicit cases; required for build")
        command.add_argument("--output", type=Path, required=True)
        if name == "build":
            command.add_argument("--size", type=int, default=200)
            command.add_argument("--seed", type=int, default=42)
    score = commands.add_parser("score")
    score.add_argument("--annotations", type=Path, required=True)
    score.add_argument("--key", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "score":
        result = score_annotations(_jsonl(args.annotations), _jsonl(args.key))
    else:
        if args.command == "build" and args.tasks_from is None:
            parser.error("build requires --tasks-from to keep classifier tuning within pilot tasks")
        task_ids = set(json.loads(args.tasks_from.read_text())["cases"]) if args.tasks_from else None
        records = list(read_records(args.input, task_ids))
        if not records:
            parser.error("No exact actions found; count-only historical probes cannot be reclassified")
        if args.command == "build":
            annotations, private, result = build_annotations(records, size=args.size, seed=args.seed)
            args.output.mkdir(parents=True, exist_ok=True)
            for name, rows in (("annotations.jsonl", annotations), ("key.jsonl", private)):
                with (args.output / name).open("x") as stream:
                    stream.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
            args.output = args.output / "manifest.json"
        else:
            result = audit(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
