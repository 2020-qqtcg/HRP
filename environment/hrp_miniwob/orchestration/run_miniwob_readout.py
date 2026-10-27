"""Local four-GPU supervisor for the fixed readout experiment.

The supervisor writes only local state and artifacts.  It intentionally has no
thread, queue, notification, or paper-editing integration.
"""

from __future__ import annotations

import fcntl
import json
import os
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime

from hrp_miniwob.adapters.native_miniwob import write_json
from hrp_miniwob.benchmarks import miniwob_readout as study


GPUS = (0, 1, 2, 3)


def gpu_idle(gpu):
    command = ["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid", "--format=csv,noheader,nounits"]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return not result.stdout.strip()


def run_checked(command, log, env=None):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as stream:
        stream.write(f"\n[{datetime.now(UTC).isoformat()}] {json.dumps(command)}\n")
        stream.flush()
        subprocess.run(command, cwd=study.ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)


def validation(experiment):
    python = sys.executable
    study.prepare_data(experiment)
    acceptance = experiment / "acceptance"
    if not (acceptance / "cpu.json").exists():
        run_checked([python, "-m", "hrp_miniwob.validation.check_miniwob_readout", "cpu", "--experiment", str(experiment)], experiment / "logs/cpu_acceptance.log")
    if not (acceptance / "gpu_preflight.json").exists():
        while not gpu_idle(0):
            write_json(experiment / "scheduler.json", {"status": "waiting_for_preflight_gpu0", "updated_at_utc": datetime.now(UTC).isoformat()})
            time.sleep(20)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"
        run_checked([python, "-m", "hrp_miniwob.validation.check_miniwob_readout", "gpu", "--experiment", str(experiment), "--device", "cuda:0"], experiment / "logs/gpu_preflight.log", env)
    study.seal(experiment)


def evaluation_jobs(experiment, phase):
    jobs = []
    if phase == "calibration":
        for model in study.MODEL_SPECS:
            jobs.append({
                "key": f"repeat/{model}",
                "output": experiment / "controls/repeat" / model,
                "completion": experiment / "controls/repeat" / model / "completion.json",
                "args": ["repeat", "--output", str(experiment / "controls/repeat" / model), "--model", model],
            })
    for model in study.MODEL_SPECS:
        for rendering in study.RENDERINGS:
            output = experiment / "evaluations" / phase / model / rendering
            jobs.append({
                "key": f"{phase}/{model}/{rendering}",
                "output": output,
                "completion": output / "completion.json",
                "args": ["evaluate", "--output", str(output), "--model", model, "--rendering", rendering, "--split", phase],
            })
    return jobs


def run_phase(experiment, phase):
    pending = [job for job in evaluation_jobs(experiment, phase) if not job["completion"].exists()]
    running, attempts = {}, {}
    python = sys.executable
    while pending or running:
        for gpu in GPUS:
            if gpu in running or not pending or not gpu_idle(gpu):
                continue
            job = pending.pop(0)
            key = job["key"]
            attempts[key] = attempts.get(key, 0) + 1
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            log = experiment / "logs" / (key.replace("/", "_") + f"_attempt{attempts[key]}.log")
            log.parent.mkdir(parents=True, exist_ok=True)
            stream = log.open("a")
            command = [python, "-m", "hrp_miniwob.benchmarks.miniwob_readout", "--experiment", str(experiment), *job["args"], "--device", "cuda:0"]
            stream.write(f"[{datetime.now(UTC).isoformat()}] {json.dumps(command)}\n")
            stream.flush()
            process = subprocess.Popen(command, cwd=study.ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            running[gpu] = {"job": job, "process": process, "stream": stream, "log": str(log), "started": time.monotonic()}

        finished = []
        for gpu, record in running.items():
            code = record["process"].poll()
            if code is None:
                continue
            record["stream"].close()
            job = record["job"]
            if code != 0 or not job["completion"].exists():
                if attempts[job["key"]] >= 3:
                    write_json(experiment / "scheduler.json", {
                        "status": "failed", "phase": phase, "job": job["key"], "exit_code": code,
                        "log": record["log"], "attempts": attempts[job["key"]],
                    })
                    raise RuntimeError(f"Fixed cell failed three times: {job['key']}")
                pending.append(job)
            finished.append(gpu)
        for gpu in finished:
            del running[gpu]

        total = len(evaluation_jobs(experiment, phase))
        complete = sum(job["completion"].exists() for job in evaluation_jobs(experiment, phase))
        write_json(experiment / "scheduler.json", {
            "status": "running", "phase": phase, "completed_phase_jobs": complete,
            "total_phase_jobs": total, "pending": len(pending),
            "running": {str(gpu): {"job": value["job"]["key"], "pid": value["process"].pid, "log": value["log"]} for gpu, value in running.items()},
            "allowed_physical_gpus": list(GPUS), "updated_at_utc": datetime.now(UTC).isoformat(),
        })
        if pending or running:
            time.sleep(10)


def main():
    experiment = study.EXPERIMENT.resolve()
    experiment.mkdir(parents=True, exist_ok=True)
    lock_path = experiment / "supervisor.lock"
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("A readout supervisor is already active") from error
        write_json(experiment / "supervisor.json", {
            "status": "running", "pid": os.getpid(), "host": socket.gethostname(),
            "started_at_utc": datetime.now(UTC).isoformat(), "allowed_physical_gpus": list(GPUS),
            "cross_task_notifications": False,
        })
        validation(experiment)
        run_phase(experiment, "calibration")
        run_phase(experiment, "test")
        study.analyze(experiment)
        run_checked([sys.executable, "-m", "hrp_miniwob.benchmarks.miniwob_readout_report"], experiment / "logs/report.log")
        write_json(experiment / "scheduler.json", {
            "status": "complete", "completed_at_utc": datetime.now(UTC).isoformat(),
            "calibration_jobs": len(evaluation_jobs(experiment, "calibration")),
            "test_jobs": len(evaluation_jobs(experiment, "test")),
        })
        write_json(experiment / "supervisor.json", {
            "status": "complete", "pid": os.getpid(), "host": socket.gethostname(),
            "completed_at_utc": datetime.now(UTC).isoformat(), "cross_task_notifications": False,
        })


if __name__ == "__main__":
    main()
