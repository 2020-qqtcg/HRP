"""Content-disjoint native generator splits and reproducible public manifests."""

import hashlib
import json
from collections import Counter, deque
from dataclasses import asdict
from pathlib import Path

from . import FAMILIES
from .browser import Browser, Task
from .model import audit_model, compile_task

SPLITS = ("search", "train", "calibration", "test", "ood")
FULL_QUOTAS = dict(zip(SPLITS, (24, 480, 96, 192, 96), strict=True))
SMOKE_QUOTAS = dict(zip(SPLITS, (1, 2, 1, 1, 1), strict=True))


def is_ood(task):
    if task.family.startswith("click-checkboxes"):
        return len(task.data["labels"]) >= 5
    return next(i for i, tile in enumerate(task.data["tiles"]) if tile["number"] == 1) >= 3


def verify_success_path(browser, model):
    """Independent native completion check for each saved generated instance."""
    paths, queue = {model.initial_state: ()}, deque([model.initial_state])
    success = None
    while queue and success is None:
        state = queue.popleft()
        for action, nxt in enumerate(model.next_state[state]):
            if model.task_rewards[state, action] == 1.:
                success = (*paths[state], action)
                break
            if nxt >= 0 and int(nxt) not in paths:
                paths[int(nxt)] = (*paths[state], action)
                queue.append(int(nxt))
    if success is None:
        raise ValueError(f"No native success path within budget: {model.task.key}")
    initial = browser.reset(model.task.seed, model.task.family)
    if initial.task != model.task:
        raise AssertionError("Task changed during verification")
    for action in success:
        result = browser.step(model.task.actions[action])
    if not result.state.terminated or result.reward != 1.:
        raise AssertionError(f"Native success disagrees with kernel: {model.task.key}")
    return len(success)


def prepare_suite(output, horizon=10, start_seed=9100000, profile="full", audit=True):
    output = Path(output)
    if (output / "manifest.json").exists():
        raise FileExistsError("Use a fresh v2 suite directory; splits are immutable")
    if horizon < 5:
        raise ValueError("Three-family suite requires at least five decisions")
    output.mkdir(parents=True, exist_ok=True)
    (output / "tasks").mkdir(exist_ok=True)
    quotas = FULL_QUOTAS if profile == "full" else SMOKE_QUOTAS
    entries, seen, family_audits = [], set(), {}
    with Browser(horizon=horizon) as browser:
        for f_index, family in enumerate(FAMILIES):
            counts = Counter()
            required = quotas
            seed = start_seed + f_index * 1000000
            scanned = 0
            while any(counts[s] < required[s] for s in SPLITS):
                if scanned >= 100000:
                    raise RuntimeError(f"Native generator exhausted scan budget for {family}: {dict(counts)}")
                task = browser.reset(seed, family).task
                seed, scanned = seed + 1, scanned + 1
                if task.fingerprint in seen:
                    continue
                splits = ("ood",) if is_ood(task) else SPLITS[:-1]
                split = next((s for s in splits if counts[s] < required[s]), None)
                if split is None:
                    continue
                model = compile_task(task, horizon)
                path_steps = verify_success_path(browser, model)
                if family not in family_audits:
                    family_audits[family] = {"task": task.key, "checks": audit_model(browser, model) if audit else 0,
                                             "scope": "all physical grading states + transition basis + final budget" if audit else "success paths only"}
                payload = json.dumps(asdict(task), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
                relative = f"tasks/{task.key}.json"
                (output / relative).write_text(payload)
                entries.append({"family": family, "seed": task.seed, "split": split, "json": relative,
                                "sha256": hashlib.sha256(payload.encode()).hexdigest(),
                                "content_sha256": task.fingerprint, "actions": len(task.actions),
                                "physical_states": task.physical_states, "verified_success_steps": path_steps})
                seen.add(task.fingerprint)
                counts[split] += 1
            print(json.dumps({"family": family, "splits": dict(counts), "scanned_seeds": scanned}), flush=True)
    manifest = {"version": "v2", "status": "complete", "profile": profile, "horizon": horizon,
                "start_seed": start_seed, "families": list(FAMILIES), "entries": entries,
                "family_browser_audits": family_audits,
                "validation": "Every saved instance has a real native success replay; representative kernel audit scope is reported separately",
                "split_unit": "deduplicated public task contents including initial observable state; no seed-only independence claim",
                "ood_rules": {"click-checkboxes": "n >= 5", "click-checkboxes-soft": "n >= 5",
                              "ascending-numbers": "initial number 1 in the two rightmost spatial IDs"}}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def load_suite(output, split):
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["version"] != "v2" or manifest["status"] != "complete":
        raise ValueError("Expected a complete v2 suite")
    if set(manifest["families"]) != set(FAMILIES):
        raise ValueError("Prepared suite does not match the current three v2 task families")
    models = []
    for entry in manifest["entries"]:
        if entry["split"] != split:
            continue
        payload = (output / entry["json"]).read_bytes()
        if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
            raise ValueError(f"Prepared v2 task changed: {entry['json']}")
        task = Task(**json.loads(payload))
        if task.fingerprint != entry["content_sha256"]:
            raise ValueError("Public task fingerprint differs from manifest")
        models.append(compile_task(task, manifest["horizon"]))
    return models
