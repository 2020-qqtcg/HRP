"""Browser-verified task splits fixed before language-model experiments."""

import hashlib
import json
import platform
import time
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np

from .assets import MANIFEST
from .browser import CheckboxBrowser, Task
from .model import TaskModel, compile_task


def reachable_states(model: TaskModel) -> np.ndarray:
    """Return sorted nonterminal states reachable under any semantic policy."""
    reachable = np.zeros(len(model.next_state), dtype=bool)
    reachable[model.initial_state] = True
    pending = [model.initial_state]
    while pending:
        state = pending.pop()
        for successor in model.next_state[state]:
            if successor >= 0 and not reachable[successor]:
                reachable[successor] = True
                pending.append(int(successor))
    return np.flatnonzero(reachable)


def prepare_suite(
    output: Path,
    horizon: int = 10,
    start_seed: int = 100,
    train_per_cell: int = 2,
    calibration_per_cell: int = 1,
    test_per_cell: int = 2,
    ood_per_cell: int = 2,
) -> dict:
    """Scan the original generator and compile disjoint, stratified task splits.

    A cell is the public pair (checkbox count, requested checkbox count).
    For n=2,3,4, successive instances fill train, calibration, then test quotas.
    All n=5,6 cells belong exclusively to the OOD split. Every state of an
    instance inherits its task split. No model behavior or oracle return is
    used for selection. A selected task whose public instruction disagrees
    with browser grading aborts preparation, rather than being replaced.
    """
    output = Path(output)
    quotas = {
        "train": train_per_cell,
        "calibration": calibration_per_cell,
        "test": test_per_cell,
        "ood": ood_per_cell,
    }
    if horizon < 1 or any(value < 0 for value in quotas.values()):
        raise ValueError("horizon must be positive and split quotas nonnegative")
    counts = {
        (n, k, split): 0
        for n in range(2, 7)
        for k in range(n + 1)
        for split in (("train", "calibration", "test") if n <= 4 else ("ood",))
    }
    requested = sum(quotas[split] for _, _, split in counts)
    output.mkdir(parents=True, exist_ok=True)
    (output / "tasks").mkdir(exist_ok=True)
    manifest = {
        "schema_version": 1,
        "status": "incomplete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "horizon": horizon,
            "start_seed": start_seed,
            "quotas_per_cell": quotas,
            "requested_tasks": requested,
        },
        "split_contract": {
            "unit": "entire_task_instance_including_all_masks_and_remaining_budgets",
            "cell": "(number_of_checkboxes, number_requested_in_public_instruction)",
            "train": "n=2,3,4; first train_per_cell seeds in each cell",
            "calibration": "n=2,3,4; next calibration_per_cell seeds in each cell",
            "test": "n=2,3,4; next test_per_cell seeds in each cell",
            "ood": "n=5,6; first ood_per_cell seeds in each cell; unseen checkbox counts",
            "iid_scope": "Within a cell, IID tests label and target-position generalization, not new task structures.",
            "sampling_distribution": "Balanced cell quotas; not the upstream generator's natural task distribution.",
            "failure_policy": "Public-information or browser-grading inconsistency aborts; no silent seed replacement.",
            "claim_scope": "One checkbox task family; sample sizes do not establish statistical power by themselves.",
        },
        "upstream": MANIFEST,
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(__file__).parent.glob("*.py"))
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "playwright": version("playwright"),
        },
        "scan_log": "scan.jsonl",
        "scanned_seeds": 0,
        "excluded_seeds": 0,
        "tasks": [],
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    started = time.perf_counter()
    seed = start_seed
    try:
        with CheckboxBrowser(horizon=horizon) as browser, (output / "scan.jsonl").open("w") as scan:
            manifest["runtime"]["chromium"] = browser.browser.version
            while len(manifest["tasks"]) < requested:
                initial = browser.reset(seed)
                task = initial.task
                n = len(task.labels)
                instruction = task.instruction
                record = {"seed": seed, "task": asdict(task), "n": n}
                manifest["scanned_seeds"] += 1
                if not instruction.startswith("Select ") or not instruction.endswith(" and click Submit."):
                    record["decision"] = "failed_public_instruction_format"
                    scan.write(json.dumps(record) + "\n")
                    scan.flush()
                    raise AssertionError(f"Seed {seed}: unrecognized public instruction")
                requested_labels = instruction[len("Select "):-len(" and click Submit.")]
                labels = [] if requested_labels == "nothing" else requested_labels.split(", ")
                if not set(labels).issubset(task.labels) or not 2 <= n <= 6:
                    record["decision"] = "failed_public_task_consistency"
                    scan.write(json.dumps(record) + "\n")
                    scan.flush()
                    raise AssertionError(f"Seed {seed}: public instruction and labels disagree")
                k = task.target_mask.bit_count()
                record["k"] = k
                candidates = ("train", "calibration", "test") if n <= 4 else ("ood",)
                split = next((name for name in candidates if counts[n, k, name] < quotas[name]), None)
                if split is None:
                    record["decision"] = "excluded_cell_quotas_filled"
                    manifest["excluded_seeds"] += 1
                else:
                    record.update({"decision": "selected_for_browser_verification", "split": split})
                scan.write(json.dumps(record) + "\n")
                scan.flush()
                if split is None:
                    seed += 1
                    continue

                print(f"Compiling {split} seed {seed}: n={n}, k={k} "
                      f"({len(manifest['tasks']) + 1}/{requested})", flush=True)
                model = compile_task(browser, seed)
                if model.task != task:
                    raise AssertionError(f"Seed {seed}: selected public task changed during compilation")
                json_path = output / "tasks" / f"seed_{seed}.json"
                npz_path = json_path.with_suffix(".npz")
                np.savez_compressed(
                    npz_path,
                    next_state=model.next_state,
                    task_rewards=model.task_rewards,
                    physical_next=model.physical_next,
                    raw_submit_rewards=model.raw_submit_rewards,
                )
                report = {
                    "task": asdict(model.task),
                    "split": split,
                    "cell": {"n": n, "k": k},
                    "horizon": model.horizon,
                    "initial_state": model.initial_state,
                    "nonterminal_states": len(model.next_state),
                    "reachable_nonterminal_states": len(reachable_states(model)),
                    "browser_checks": model.browser_checks,
                    "verification": "all_physical_masks_and_actions_at_ordinary_and_last_budget",
                    "public_instruction_matches_browser_grading": True,
                    "upstream_revision": MANIFEST["revision"],
                }
                json_path.write_text(json.dumps(report, indent=2) + "\n")
                manifest["tasks"].append({
                    "seed": seed,
                    "n": n,
                    "k": k,
                    "split": split,
                    "json": str(json_path.relative_to(output)),
                    "npz": str(npz_path.relative_to(output)),
                    "sha256": {
                        "json": hashlib.sha256(json_path.read_bytes()).hexdigest(),
                        "npz": hashlib.sha256(npz_path.read_bytes()).hexdigest(),
                    },
                })
                counts[n, k, split] += 1
                manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
                seed += 1
        manifest["scan_sha256"] = hashlib.sha256((output / "scan.jsonl").read_bytes()).hexdigest()
        manifest["cell_counts"] = [
            {"n": n, "k": k, "split": split, "count": count}
            for (n, k, split), count in counts.items()
        ]
        manifest["status"] = "complete"
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = {"seed": seed, "type": type(error).__name__, "message": str(error)}
        raise
    finally:
        manifest["elapsed_seconds"] = time.perf_counter() - started
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_suite(output: Path, split: str) -> list[TaskModel]:
    """Load one completed split and verify the saved browser-compiled artifacts."""
    if split not in {"train", "calibration", "test", "ood"}:
        raise ValueError("split must be train, calibration, test, or ood")
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Only a complete browser-verified suite can be loaded")
    seeds = [entry["seed"] for entry in manifest["tasks"]]
    if len(seeds) != len(set(seeds)):
        raise ValueError("A task seed appears more than once across suite splits")
    models = []
    for entry in manifest["tasks"]:
        if entry["split"] != split:
            continue
        for suffix in ("json", "npz"):
            path = output / entry[suffix]
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"][suffix]:
                raise ValueError(f"Suite artifact checksum mismatch: {path}")
        report = json.loads((output / entry["json"]).read_text())
        if report["split"] != split or report["task"]["seed"] != entry["seed"]:
            raise ValueError("Task identity or split disagrees with suite manifest")
        task = Task(report["task"]["seed"], report["task"]["instruction"],
                    tuple(report["task"]["labels"]))
        with np.load(output / entry["npz"], allow_pickle=False) as arrays:
            models.append(TaskModel(
                task=task,
                horizon=report["horizon"],
                next_state=arrays["next_state"],
                task_rewards=arrays["task_rewards"],
                physical_next=arrays["physical_next"],
                raw_submit_rewards=arrays["raw_submit_rewards"],
                browser_checks=report["browser_checks"],
            ))
    return models
