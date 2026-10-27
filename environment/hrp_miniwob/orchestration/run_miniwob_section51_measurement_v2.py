"""Run the held-out probability cells for Section 5.1 measurement V2."""

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
from hrp_miniwob.benchmarks import miniwob_section51 as study
from hrp_miniwob.benchmarks import miniwob_section51_measurement_v2 as v2
from hrp_miniwob.orchestration.run_miniwob_section51 import free_gpus


def alive(job):
    try:
        process = psutil.Process(job["pid"])
        return (
            process.create_time() == job["process_created"]
            and process.status() != psutil.STATUS_ZOMBIE
            and process.cmdline() == job["command"]
        )
    except (psutil.NoSuchProcess, psutil.AccessDenied, KeyError):
        return False


def jobs(experiment):
    base = [
        sys.executable, "-m", "hrp_miniwob.benchmarks.miniwob_section51", "measure-cell",
        "--experiment", str(experiment), "--split", "test",
    ]
    return {
        cell: base + ["--cell", cell, "--device", "cuda:0"]
        for cell in v2.TEST_CELLS
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=v2.EXPERIMENT)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    experiment = args.experiment.resolve()
    commands = jobs(experiment)
    if not args.execute:
        print(json.dumps({"cells": list(commands), "gpus": list(study.PHYSICAL_GPUS)}, indent=2))
        return
    frozen = v2.read(experiment / "calibration/frozen_choices.json")
    if frozen["status"] != "frozen":
        raise RuntimeError("Calibration choices are not frozen")
    manifest = v2.read(experiment / "suite/manifest.json")
    if manifest["status"] != "complete" or len(manifest["entries"]) != 576:
        raise RuntimeError("Fresh held-out suite is incomplete")
    logs = experiment / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    scheduler_path = experiment / "scheduler.json"
    state = v2.read(scheduler_path) if scheduler_path.exists() else {
        "status": "starting", "jobs": {}, "created_at_utc": datetime.now(UTC).isoformat()
    }
    children = {}
    while not all(study.cell_complete(experiment, "test", cell) for cell in commands):
        failures = []
        for cell, record in state["jobs"].items():
            if record.get("status") != "running":
                continue
            finished = children[cell].poll() is not None if cell in children else not alive(record)
            if finished:
                complete = study.cell_complete(experiment, "test", cell)
                record.update(
                    status="complete" if complete else "failed",
                    finished_at_utc=datetime.now(UTC).isoformat(),
                    wall_seconds=time.time() - record["started_epoch"],
                )
                if not complete:
                    failures.append(cell)
        if failures:
            state.update(status="failed", failed_jobs=failures)
            write_json(scheduler_path, state)
            raise RuntimeError(f"Probability cells failed: {failures}")
        occupied = {
            record["gpu"] for record in state["jobs"].values()
            if record.get("status") == "running"
        }
        available = [gpu for gpu in free_gpus() if gpu in study.PHYSICAL_GPUS and gpu not in occupied]
        for cell in sorted(commands, key=lambda name: ("base" not in name, "H3" in name, name)):
            if study.cell_complete(experiment, "test", cell):
                state["jobs"].setdefault(cell, {"status": "complete", "reused": True})
                continue
            previous = state["jobs"].get(cell)
            if previous and previous.get("status") == "running":
                continue
            if not available:
                break
            gpu = available.pop(0)
            attempts = previous.get("attempts", 0) + 1 if previous else 1
            log = logs / f"test_{cell}_attempt{attempts}.log"
            env = os.environ | {
                "CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false",
            }
            stream = log.open("a")
            child = subprocess.Popen(
                commands[cell], cwd=study.ROOT, env=env, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True,
            )
            state["jobs"][cell] = {
                "status": "running", "pid": child.pid,
                "process_created": psutil.Process(child.pid).create_time(),
                "gpu": gpu, "command": commands[cell], "log": str(log),
                "attempts": attempts, "started_epoch": time.time(),
                "started_at_utc": datetime.now(UTC).isoformat(),
            }
            children[cell] = child
        completed = sum(study.cell_complete(experiment, "test", cell) for cell in commands)
        state.update(
            status="test_probabilities", completed_cells=completed, total_cells=len(commands),
            updated_at_utc=datetime.now(UTC).isoformat(),
        )
        write_json(scheduler_path, state)
        time.sleep(15)
    state.update(
        status="probabilities_complete", completed_cells=len(commands), total_cells=len(commands),
        completed_at_utc=datetime.now(UTC).isoformat(),
    )
    write_json(scheduler_path, state)
    write_json(experiment / "progress.json", {
        "status": "test_probabilities_complete", "completed_cells": len(commands),
        "updated_at_utc": datetime.now(UTC).isoformat(),
    })
    print(json.dumps({"status": state["status"], "cells": len(commands)}))


if __name__ == "__main__":
    main()
