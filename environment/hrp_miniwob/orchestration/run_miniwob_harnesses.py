"""Print the H0-H5 training/cross-evaluation plan; opt in to execution."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from hrp_miniwob.harnesses.miniwob_harnesses import PROFILES
from hrp_miniwob.benchmarks.miniwob_harnesses import DEFAULT_EXPERIMENT, bindings, read_config
from hrp_miniwob.benchmarks.miniwob_v2 import require_v2_path
from hrp_miniwob.benchmarks.miniwob_v2_train import resolve_budget


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--harnesses", choices=PROFILES, nargs="+", default=list(PROFILES))
    parser.add_argument("--methods", choices=("exactpg",), nargs="+", default=["exactpg"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--updates", type=int)
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument("--gpu", type=int, choices=range(4))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if min(args.seeds) < 0 or any(len(values) != len(set(values)) for values in (args.seeds, args.harnesses, args.methods)):
        parser.error("Use distinct harnesses/methods and distinct nonnegative seeds")
    if args.execute and args.gpu is None:
        parser.error("--execute requires a physical --gpu")
    experiment = require_v2_path(args.experiment)
    command = [sys.executable, "-m", "hrp_miniwob.benchmarks.miniwob_harnesses"]
    jobs = []
    for method in args.methods:
        budget = argparse.Namespace(**(vars(args) | {"method": method}))
        try:
            resolve_budget(budget)
        except ValueError as error:
            parser.error(str(error))
        for seed in args.seeds:
            runs = []
            for harness in args.harnesses:
                run = experiment / "runs" / f"{method}_{harness}_seed{seed}"
                runs.append(run)
                train = command + ["train", "--experiment", str(experiment), "--output", str(run),
                                   "--harness", harness, "--method", method, "--seed", str(seed),
                                   "--batch-size", str(budget.batch_size), "--updates", str(budget.updates),
                                   "--checkpoint-interval", str(budget.checkpoint_interval), "--resume"]
                jobs.append({"stage": run.name, "command": train})
            for split in ("test", "ood"):
                stage = f"cross_{method}_seed{seed}_{split}"
                jobs.append({"stage": stage, "command": command + ["cross-eval", "--experiment", str(experiment),
                             "--output", str(experiment / stage), "--harnesses", *args.harnesses,
                             "--train-runs", *map(str, runs), "--split", split]})
    print(json.dumps({"harness_protocol": "h0_h5_v2", "jobs": jobs,
                      "execution": "sequential single-GPU jobs; serial browser sampling; no training without --execute"}, indent=2))
    if not args.execute:
        return
    with bindings():
        read_config(experiment)
    env = os.environ | {"CUDA_VISIBLE_DEVICES": str(args.gpu), "OMP_NUM_THREADS": "2",
                        "PYTHONDONTWRITEBYTECODE": "1", "TOKENIZERS_PARALLELISM": "false"}
    logs = experiment / "pipeline_logs"
    logs.mkdir(exist_ok=True)
    for job in jobs:
        print(json.dumps({"starting": job["stage"]}), flush=True)
        with (logs / f"{job['stage']}.log").open("a") as stream:
            subprocess.run(job["command"], cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)


if __name__ == "__main__":
    main()
