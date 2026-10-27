"""Pinned local Terminal-Bench datasets shared by image preparation and evaluation."""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from harbor.models.task.task import Task

ROOT = Path(__file__).resolve().parents[2]


def checkout_terminal_datasets(config: dict, benchmarks: list[str]) -> None:
    """Populate missing downloads without changing an existing checkout."""
    for benchmark in benchmarks:
        spec = config["datasets"][benchmark]
        destination = ROOT / spec["path"]
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", "--no-checkout", spec["repository"], str(destination)], check=True)
            subprocess.run(["git", "-C", str(destination), "checkout", "--detach", spec["commit"]], check=True)


def task_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    for entry in sorted(path.rglob("*")):
        if entry.is_file() and "__pycache__" not in entry.parts:
            digest.update(str(entry.relative_to(path)).encode()+b"\0"+entry.read_bytes()+b"\0")
    return digest.hexdigest()


def load_terminal_tasks(config: dict, benchmark: str) -> dict[str, dict]:
    spec = config["datasets"][benchmark]
    repository = ROOT / spec["path"]
    commit = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    if commit != spec["commit"]:
        raise ValueError(f"{benchmark}: checkout {spec['commit']} before preparing or evaluating")
    dirty = subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain"], text=True)
    if dirty.strip():
        raise ValueError(f"{benchmark}: pinned task repository has uncommitted changes")
    root = repository / spec["tasks_dir"]
    files = sorted(root.glob("*/task.toml"))
    if len(files) != spec["task_count"]:
        raise ValueError(f"{benchmark}: expected {spec['task_count']} tasks, found {len(files)}")
    requested = config.get("task_ids", {}).get(benchmark)
    if requested is not None:
        if config.get("case_limit") is not None:
            raise ValueError("task_ids and case_limit cannot be combined")
        if len(requested) != len(set(requested)):
            raise ValueError(f"{benchmark}: duplicate task_ids")
        by_id = {file.parent.name: file for file in files}
        missing = [task_id for task_id in requested if task_id not in by_id]
        if missing:
            raise ValueError(f"{benchmark}: unknown task_ids: {missing}")
        files = [by_id[task_id] for task_id in requested]
    limit = config.get("case_limit")
    if limit is not None:
        files = files[:limit]
    tasks = {}
    for file in files:
        task = Task(file.parent)
        tasks[file.parent.name] = {"path": str(file.parent.resolve()),
            "task_fingerprint": task_fingerprint(file.parent),
            "source_image": task.config.environment.docker_image,
            "agent_timeout_sec": task.config.agent.timeout_sec,
            "verifier_timeout_sec": task.config.verifier.timeout_sec}
    return tasks
