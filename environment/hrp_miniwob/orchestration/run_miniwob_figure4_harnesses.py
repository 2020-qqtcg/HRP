"""Schedule Figure 4 native H0--H5 probability cells on physical GPUs 0--3."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import psutil

from hrp_miniwob.adapters.native_miniwob import write_json
from hrp_miniwob.benchmarks import miniwob_figure4_harnesses as study
from hrp_miniwob.orchestration.run_miniwob_section51 import free_gpus


def alive(job):
    try:
        process = psutil.Process(job["pid"])
        return (process.create_time() == job["process_created"]
                and process.status() != psutil.STATUS_ZOMBIE
                and process.cmdline() == job["command"])
    except (psutil.NoSuchProcess, psutil.AccessDenied, KeyError):
        return False


def command(experiment, split, harness):
    return [
        sys.executable, "-m", "hrp_miniwob.benchmarks.miniwob_figure4_harnesses",
        "--experiment", str(experiment), "measure-cell", "--split", split,
        "--harness", harness, "--device", "cuda:0",
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=study.EXPERIMENT)
    parser.add_argument("--split", choices=("calibration", "test"), required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    experiment = args.experiment.resolve()
    commands = {harness: command(experiment, args.split, harness) for harness in study.HARNESSES}
    if not args.execute:
        print(json.dumps(commands, indent=2))
        return
    logs = experiment / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    scheduler_path = experiment / f"scheduler_{args.split}.json"
    state = study.read(scheduler_path) if scheduler_path.exists() else {
        "status": "starting", "split": args.split, "jobs": {},
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    children = {}
    while not all(study.cell_complete(experiment, args.split, harness) for harness in study.HARNESSES):
        failures = []
        for harness, record in state["jobs"].items():
            if record.get("status") != "running":
                continue
            finished = children[harness].poll() is not None if harness in children else not alive(record)
            if finished:
                complete = study.cell_complete(experiment, args.split, harness)
                record.update(
                    status="complete" if complete else "failed",
                    finished_at_utc=datetime.now(UTC).isoformat(),
                    wall_seconds=time.time() - record["started_epoch"],
                )
                if not complete:
                    failures.append(harness)
        if failures:
            state.update(status="failed", failed_jobs=failures, updated_at_utc=datetime.now(UTC).isoformat())
            write_json(scheduler_path, state)
            raise RuntimeError(f"Figure 4 probability cells failed: {failures}")
        occupied = {record["gpu"] for record in state["jobs"].values()
                    if record.get("status") == "running" and record.get("gpu") is not None}
        available = [gpu for gpu in free_gpus() if gpu in study.PHYSICAL_GPUS and gpu not in occupied]
        cpu_running = sum(record.get("status") == "running" and record.get("gpu") is None
                          for record in state["jobs"].values())
        for harness in study.HARNESSES:
            if study.cell_complete(experiment, args.split, harness):
                state["jobs"].setdefault(harness, {"status": "complete", "reused": True})
                continue
            previous = state["jobs"].get(harness)
            if previous and previous.get("status") == "running":
                continue
            imported = harness in {"H0", "H3"}
            if imported and cpu_running >= 2:
                continue
            if not imported and not available:
                continue
            gpu = None if imported else available.pop(0)
            cpu_running += int(imported)
            attempts = previous.get("attempts", 0) + 1 if previous else 1
            log = logs / f"{args.split}_{harness}_attempt{attempts}.log"
            env = os.environ | {
                "CUDA_VISIBLE_DEVICES": "" if gpu is None else str(gpu),
                "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                "TOKENIZERS_PARALLELISM": "false",
            }
            stream = log.open("a")
            child = subprocess.Popen(
                commands[harness], cwd=study.ROOT, env=env, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True,
            )
            state["jobs"][harness] = {
                "status": "running", "pid": child.pid,
                "process_created": psutil.Process(child.pid).create_time(),
                "gpu": gpu, "command": commands[harness], "log": str(log),
                "attempts": attempts, "started_epoch": time.time(),
                "started_at_utc": datetime.now(UTC).isoformat(),
            }
            children[harness] = child
        completed = sum(study.cell_complete(experiment, args.split, harness) for harness in study.HARNESSES)
        total_completed = sum(study.cell_complete(experiment, split, harness)
                              for split in ("calibration", "test") for harness in study.HARNESSES)
        state.update(status="probabilities", completed_cells=completed, total_cells=6,
                     updated_at_utc=datetime.now(UTC).isoformat())
        write_json(scheduler_path, state)
        write_json(experiment / "progress.json", {
            "status": f"{args.split}_probabilities", "completed_cells": total_completed,
            "total_cells": 12, "active_processes": [record["pid"] for record in state["jobs"].values()
                                                     if record.get("status") == "running"],
            "updated_at_utc": datetime.now(UTC).isoformat(),
        })
        time.sleep(15)
    state.update(status="complete", completed_cells=6, total_cells=6,
                 completed_at_utc=datetime.now(UTC).isoformat())
    write_json(scheduler_path, state)
    print(json.dumps({"status": "complete", "split": args.split, "cells": 6}))


if __name__ == "__main__":
    main()
