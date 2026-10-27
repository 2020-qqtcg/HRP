"""Run the fixed Section 5.1 experiment on physical GPUs 4, 5, 6 and 7."""

import argparse
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path

from hrp_miniwob.benchmarks import miniwob_section51_self_reference as study


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=study.EXPERIMENT)
    args = parser.parse_args(argv)
    experiment = args.experiment.resolve()
    manifest = study.verify_sources(experiment)
    for kind in ("cpu", "gpu"):
        if study.read(experiment / "acceptance" / f"{kind}.json")["status"] != "passed":
            raise RuntimeError("CPU and GPU acceptance must pass before submission")
    lock = (experiment / "scheduler.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    devices = study.gpu_inventory()
    for gpu in study.GPUS:
        if devices[str(gpu)]["used_mib_at_prepare"] > 1500:
            raise RuntimeError(f"Physical GPU {gpu} is occupied; no other process will be stopped")
    snapshot = experiment / "source_snapshot"
    env = os.environ | {"PYTHONPATH": f"{snapshot / 'environment'}:{snapshot}",
                        "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                        "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1",
                        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                        "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    logs = experiment / "logs"
    logs.mkdir(exist_ok=True)
    state = {"status": "starting", "scheduler_pid": os.getpid(), "started_at_utc": study.now(),
             "physical_gpus": list(study.GPUS), "total_evaluations": 3456, "jobs": {}}
    children = {}
    try:
        for shard, gpu in enumerate(study.GPUS):
            command = [sys.executable, "-B", "-m", "hrp_miniwob.benchmarks.miniwob_section51_self_reference",
                       "--experiment", str(experiment), "worker", "--shard", str(shard)]
            worker_env = env | {"CUDA_VISIBLE_DEVICES": manifest["gpu_inventory"][str(gpu)]["uuid"]}
            with (logs / f"gpu{gpu}.log").open("a") as stream:
                child = subprocess.Popen(command, cwd=snapshot, env=worker_env, stdin=subprocess.DEVNULL,
                                         stdout=stream, stderr=subprocess.STDOUT)
            children[gpu] = child
            state["jobs"][str(gpu)] = {"pid": child.pid, "gpu": gpu, "shard": shard,
                                        "log": str(logs / f"gpu{gpu}.log"), "command": command}
        state.update(status="running")
        study.write(experiment / "scheduler.json", state)
        while True:
            completed, done, failures = 0, 0, []
            for gpu, child in children.items():
                code = child.poll()
                if code is not None:
                    done += 1
                    if code:
                        failures.append({"gpu": gpu, "exit_code": code})
                path = experiment / "workers" / f"gpu{gpu}.json"
                if path.exists():
                    completed += study.read(path).get("completed_evaluations", 0)
            study.write(experiment / "progress.json", {
                "status": "running" if not failures else "failed", "completed_evaluations": completed,
                "total_evaluations": 3456, "finished_workers": done,
                "failures": failures, "updated_at_utc": study.now(),
            })
            if failures:
                # Stop only children owned by THIS submission, retaining all complete task files.
                for child in children.values():
                    if child.poll() is None:
                        child.terminate()
                raise RuntimeError(f"Section 5.1 worker failure: {failures}")
            if done == len(children):
                break
            time.sleep(15)
        state.update(status="analyzing")
        study.write(experiment / "scheduler.json", state)
        command = [sys.executable, "-B", "-m", "hrp_miniwob.benchmarks.miniwob_section51_self_reference",
                   "--experiment", str(experiment), "analyze"]
        subprocess.run(command, cwd=snapshot, env=env | {"CUDA_VISIBLE_DEVICES": ""}, check=True)
        state.update(status="complete", completed_at_utc=study.now())
        study.write(experiment / "scheduler.json", state)
        study.write(experiment / "progress.json", {"status": "complete", "completed_evaluations": 3456,
                                                    "total_evaluations": 3456, "completed_at_utc": study.now()})
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}", updated_at_utc=study.now())
        study.write(experiment / "scheduler.json", state)
        study.write(experiment / "progress.json", state)
        raise


if __name__ == "__main__":
    main()
