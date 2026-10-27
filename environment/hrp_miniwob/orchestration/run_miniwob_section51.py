"""Resumable four-GPU supervisor for the fixed Section 5.1 experiment."""

from __future__ import annotations

import argparse
import fcntl
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
from hrp_miniwob.benchmarks.miniwob_meta_harness import digest

GPUS = (0, 1, 2, 3)


def read(path):
    return json.loads(Path(path).read_text())


def free_gpus():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=True,
    )
    return [
        int(line.split(",")[0]) for line in result.stdout.splitlines()
        if int(line.split(",")[0]) in GPUS and int(line.split(",")[1]) < 512
    ]


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


def cell_jobs(experiment, split):
    base = [
        sys.executable, "-m", "hrp_miniwob.benchmarks.miniwob_section51", "measure-cell",
        "--experiment", str(experiment), "--split", split,
    ]
    return {
        f"{split}_{cell}": {
            "split": split,
            "cell": cell,
            "command": base + ["--cell", cell, "--device", "cuda:0"],
        }
        for cell in study.cell_specs(split)
    }


def cell_done(experiment, job):
    return study.cell_complete(experiment, job["split"], job["cell"])


def run_cpu(command, experiment, log):
    args = [
        sys.executable, "-m", "hrp_miniwob.benchmarks.miniwob_section51_analysis",
        command, "--experiment", str(experiment),
    ]
    started = time.monotonic()
    with log.open("a") as stream:
        result = subprocess.run(
            args, cwd=study.ROOT,
            env=os.environ | {"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"},
            stdout=stream, stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(f"CPU phase failed: {command}; see {log}")
    return time.monotonic() - started


def run_cell_phase(experiment, split, state, logs):
    jobs = cell_jobs(experiment, split)
    children = {}
    phase_key = f"{split}_jobs"
    state.setdefault(phase_key, {})
    while not all(cell_done(experiment, spec) for spec in jobs.values()):
        failures = []
        for key, job in state[phase_key].items():
            if job["status"] != "running":
                continue
            finished = children[key].poll() is not None if key in children else not alive(job)
            if finished:
                complete = cell_done(experiment, jobs[key])
                job.update(
                    status="complete" if complete else "failed",
                    finished_at_utc=datetime.now(UTC).isoformat(),
                    wall_seconds=time.time() - job["started_epoch"],
                )
                if not complete:
                    failures.append(key)
        if failures:
            state.update(status="failed", failed_jobs=failures, updated_at_utc=datetime.now(UTC).isoformat())
            write_json(experiment / "scheduler.json", state)
            raise RuntimeError(f"GPU probability cells failed: {failures}")

        occupied = {
            job["gpu"] for job in state[phase_key].values() if job.get("status") == "running"
        }
        available = [gpu for gpu in free_gpus() if gpu not in occupied]
        for key in sorted(jobs, key=lambda name: ("base" not in name, "H3" in name, name)):
            spec = jobs[key]
            if cell_done(experiment, spec):
                state[phase_key].setdefault(key, {"status": "complete", "reused": True})
                continue
            old = state[phase_key].get(key)
            if old and old.get("status") == "running":
                continue
            if not available:
                break
            gpu = available.pop(0)
            log_path = logs / f"{key}_attempt{1 + len(old.get('previous_attempts', [])) if old else 1}.log"
            env = os.environ | {
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "OMP_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "2",
                "TOKENIZERS_PARALLELISM": "false",
            }
            with log_path.open("a") as stream:
                child = subprocess.Popen(
                    spec["command"], cwd=study.ROOT, env=env,
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                )
            attempts = []
            if old:
                attempts = old.get("previous_attempts", []) + [
                    {k: v for k, v in old.items() if k != "previous_attempts"}
                ]
            state[phase_key][key] = {
                "status": "running",
                "pid": child.pid,
                "process_created": psutil.Process(child.pid).create_time(),
                "gpu": gpu,
                "command": spec["command"],
                "started_epoch": time.time(),
                "started_at_utc": datetime.now(UTC).isoformat(),
                "log": str(log_path),
                "previous_attempts": attempts,
            }
            children[key] = child
            print(json.dumps({"started": key, "gpu": gpu, "pid": child.pid}), flush=True)
        completed = sum(cell_done(experiment, spec) for spec in jobs.values())
        state.update(
            status=f"{split}_probabilities",
            phase=split,
            completed_cells=completed,
            total_cells=len(jobs),
            updated_at_utc=datetime.now(UTC).isoformat(),
        )
        write_json(experiment / "scheduler.json", state)
        time.sleep(15)
    state[f"{split}_completed_at_utc"] = datetime.now(UTC).isoformat()
    write_json(experiment / "scheduler.json", state)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=study.EXPERIMENT)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    experiment = args.experiment.resolve()
    if not args.execute:
        print(json.dumps({
            "gpus": GPUS,
            "calibration_cells": list(study.cell_specs("calibration")),
            "test_cells": list(study.cell_specs("test")),
            "phases": ["calibration GPU", "B/C/D calibration", "seal", "test GPU", "predict", "seal predictions", "score", "report"],
        }, indent=2))
        return
    if read(experiment / "acceptance/cpu.json")["status"] != "passed":
        raise RuntimeError("CPU acceptance has not passed")
    if read(experiment / "acceptance/gpu.json")["status"] != "passed":
        raise RuntimeError("GPU acceptance has not passed")
    logs = experiment / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    lock_path = experiment / ".scheduler.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = experiment / "scheduler.json"
        state = read(state_path) if state_path.exists() else {
            "status": "starting",
            "started_at_utc": datetime.now(UTC).isoformat(),
            "physical_gpus": list(GPUS),
            "external_notifications": False,
        }
        state["supervisor_pid"] = os.getpid()
        write_json(state_path, state)

        run_cell_phase(experiment, "calibration", state, logs)
        calibration_commands = (
            ("calibrate-b", experiment / "calibration/B/selection.json"),
            ("calibrate-c", experiment / "calibration/C/calibration.json"),
            ("calibrate-d", experiment / "calibration/D/calibration.json"),
            ("freeze-calibration", experiment / "calibration/frozen_choices.json"),
        )
        state.setdefault("cpu_phases", {})
        for command, marker in calibration_commands:
            if not marker.exists():
                state.update(status=command, updated_at_utc=datetime.now(UTC).isoformat())
                write_json(state_path, state)
                wall = run_cpu(command, experiment, logs / f"{command}.log")
                state["cpu_phases"][command] = {"status": "complete", "wall_seconds": wall, "marker_sha256": digest(marker)}
                write_json(state_path, state)
        if not (experiment / "config.json").exists():
            state.update(status="seal", updated_at_utc=datetime.now(UTC).isoformat())
            write_json(state_path, state)
            with (logs / "seal.log").open("a") as stream:
                result = subprocess.run(
                    [sys.executable, "-m", "hrp_miniwob.benchmarks.miniwob_section51", "seal", "--experiment", str(experiment)],
                    cwd=study.ROOT, stdout=stream, stderr=subprocess.STDOUT,
                    check=False,
                )
            if result.returncode:
                raise RuntimeError("Experiment sealing failed")
        study.verify_seal(experiment)

        run_cell_phase(experiment, "test", state, logs)
        for command, marker in (
            ("predict-c", experiment / "predictions/C/predictions.json"),
            ("predict-d", experiment / "predictions/D/predictions.json"),
        ):
            if not marker.exists():
                state.update(status=command, updated_at_utc=datetime.now(UTC).isoformat())
                write_json(state_path, state)
                wall = run_cpu(command, experiment, logs / f"{command}.log")
                state["cpu_phases"][command] = {"status": "complete", "wall_seconds": wall, "marker_sha256": digest(marker)}
                write_json(state_path, state)
        seal_predictions = experiment / "predictions/seal.json"
        if not seal_predictions.exists():
            wall = run_cpu("seal-predictions", experiment, logs / "seal-predictions.log")
            state["cpu_phases"]["seal-predictions"] = {"status": "complete", "wall_seconds": wall, "marker_sha256": digest(seal_predictions)}
            write_json(state_path, state)
        for command, marker in (
            ("score-c", experiment / "scores/C/scores.json"),
            ("score-d", experiment / "scores/D/scores.json"),
            ("report", experiment / "analysis/completion.json"),
        ):
            if not marker.exists():
                state.update(status=command, updated_at_utc=datetime.now(UTC).isoformat())
                write_json(state_path, state)
                wall = run_cpu(command, experiment, logs / f"{command}.log")
                state["cpu_phases"][command] = {"status": "complete", "wall_seconds": wall, "marker_sha256": digest(marker)}
                write_json(state_path, state)
        state.update(
            status="complete",
            completed_at_utc=datetime.now(UTC).isoformat(),
            report=str(study.DOCS / "REPORT.md"),
            analysis_sha256=digest(experiment / "analysis/diagnostics_data.json"),
        )
        write_json(state_path, state)
        print(json.dumps({"status": "complete", "report": state["report"]}), flush=True)


if __name__ == "__main__":
    main()
