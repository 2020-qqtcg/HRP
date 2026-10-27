"""Task-averaged Exact-PG training with fixed microbatches and resumable checkpoints."""

import json
import os
import shutil
import socket
import time
from datetime import UTC, datetime

import numpy as np
import torch

from hrp_miniwob.adapters.miniwob_v2_batches import (
    MICROBATCH_STATES,
    exact_batch_update,
)
from hrp_miniwob.adapters.native_miniwob import write_json
from hrp_miniwob.benchmarks import miniwob_meta_harness as host


def resolve_budget(args):
    """Batch size is tasks per Exact-PG optimizer update."""
    if args.batch_size is None:
        args.batch_size = 4
    if args.updates is None:
        args.updates = 1024
    if args.checkpoint_interval is None:
        args.checkpoint_interval = 64
    if min(args.updates, args.checkpoint_interval, args.batch_size) < 1:
        raise ValueError("Batch size and update budgets must be positive")
    settings = {"batch_size": args.batch_size, "forward_microbatch_states": MICROBATCH_STATES}
    args.rollout_episodes = None
    return settings


def train(args):
    """Run inside miniwob_v2.bindings(training=True)."""
    from hrp_miniwob.adapters.verl_miniwob import make_worker

    settings = resolve_budget(args)
    nodes = getattr(args, "checkpoint_updates", None)
    if nodes is not None:
        if not nodes or nodes != sorted(set(nodes)) or nodes[0] != 0 or nodes[-1] != args.updates:
            raise ValueError("Explicit checkpoint updates must increase from zero to the final budget")
    experiment, output = args.experiment.resolve(), args.output.resolve()
    config, candidate = host.read_config(experiment), host.Candidate(args.candidate)
    if not args.device.startswith("cuda"):
        raise ValueError("The training worker requires CUDA")
    increment = 1
    output.mkdir(parents=True, exist_ok=True)
    identity = {"config_sha256": host.digest(experiment / "config.json"), "candidate_sha256": candidate.sha256,
                "method": args.method, "seed": args.seed, "updates": args.updates,
                "checkpoint_interval": args.checkpoint_interval, "rollout_episodes": args.rollout_episodes,
                "limit": args.limit, "source_sha256": host.source_hashes(), "batch": settings}
    initial_adapter = getattr(args, "initial_adapter", None)
    if initial_adapter is not None:
        initial_adapter = initial_adapter.resolve()
        identity["initialization"] = {"adapter": str(initial_adapter), "optimizer": "reset",
            "weights_sha256": host.digest(initial_adapter / "adapter_model.safetensors"),
            "config_sha256": host.digest(initial_adapter / "adapter_config.json")}
    if nodes is not None:
        identity["checkpoint_updates"] = nodes
    manifest_path = output / "run.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["identity"] != identity:
            raise ValueError("Resume settings, batch sizes, data, model, candidate or source changed")
        if manifest["status"] == "complete":
            return
        if not args.resume:
            raise ValueError("Use --resume for interrupted training")
    else:
        manifest = {"identity": identity, "status": "initializing", "checkpoints": [], "accounting": {},
                    "created_at_utc": datetime.now(UTC).isoformat(), "candidate": str(output / "candidate.py")}
        shutil.copyfile(candidate.path, output / "candidate.py")
    attempt = {"started_at_utc": datetime.now(UTC).isoformat(), "pid": os.getpid(), "accounting": {},
               "device": args.device, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    manifest.setdefault("attempts", []).append(attempt)
    attempt_index = len(manifest["attempts"]) - 1
    write_json(manifest_path, manifest)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    os.environ.update(WORLD_SIZE="1", RANK="0", LOCAL_WORLD_SIZE="1", LOCAL_RANK="0",
                      MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    actor = adapter = None
    started = time.monotonic()
    try:
        actor = make_worker(config["actor_model"], "actor", args.device, args.seed, config["actor_lr"], config["rank"])
        if initial_adapter is not None and not manifest["checkpoints"]:
            actor.engine.initialize_adapter(initial_adapter)
        frozen = host.load_frozen(config, args.device)
        adapter = host.adapter_for(actor.engine.policy, candidate, frozen, config, experiment)
        for worker in (actor,):
            worker.engine.policy.metadata["effective_batch"] = settings
            worker.engine.policy.metadata["optimizer"] = (
                f"Exact-PG; mean of {args.batch_size} task gradients")
        tasks = host.models_for(experiment, "train", args.limit)
        calibration = host.models_for(experiment, "calibration", args.limit)
        task_rng = np.random.default_rng(np.random.SeedSequence(args.seed, spawn_key=(10,)))
        action_rng = np.random.default_rng(np.random.SeedSequence(args.seed, spawn_key=(11,)))
        completed, episodes, task_draws = 0, 0, 0

        def save():
            directory = output / f"checkpoint_{actor.engine.updates:06d}"
            actor.save_checkpoint(str(directory / "actor"), global_step=actor.engine.updates)
            write_json(directory / "controller.json", {"task_rng": task_rng.bit_generator.state,
                       "action_rng": action_rng.bit_generator.state, "episodes": episodes,
                       "task_draws": task_draws, "accounting": manifest["accounting"]})
            torch.save({"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}, directory / "rng.pt")
            entry = {"updates": actor.engine.updates, "episodes": episodes, "task_draws": task_draws,
                     "directory": directory.name,
                     "files_sha256": {str(p.relative_to(directory)): host.digest(p) for p in directory.rglob("*") if p.is_file()}}
            manifest["checkpoints"].append(entry)
            write_json(manifest_path, manifest)
            return entry

        def calibrate(entry):
            result = host.evaluate_models(calibration, adapter, output / entry["directory"] / "calibration")
            entry["calibration"] = result["metrics"]
            write_json(manifest_path, manifest)

        if manifest["checkpoints"]:
            entry = manifest["checkpoints"][-1]
            directory = output / entry["directory"]
            for name, expected in entry["files_sha256"].items():
                if host.digest(directory / name) != expected:
                    raise ValueError(f"Checkpoint changed: {name}")
            actor.load_checkpoint(str(directory / "actor"), del_local_after_load=False)
            state = json.loads((directory / "controller.json").read_text())
            task_rng.bit_generator.state, action_rng.bit_generator.state = state["task_rng"], state["action_rng"]
            episodes, task_draws = state["episodes"], state["task_draws"]
            manifest["accounting"] = state["accounting"]
            rng = torch.load(directory / "rng.pt", map_location="cpu", weights_only=True)
            torch.set_rng_state(rng["cpu"])
            torch.cuda.set_rng_state_all(rng["cuda"])
            completed = actor.engine.updates
            attempt["resumed_from_updates"] = completed
        else:
            entry = save()
        if "calibration" not in entry:
            calibrate(entry)
        while completed < args.updates:
            tick = time.monotonic()
            before = {"actor_forwards": actor.engine.policy.forward_observations,
                      "actor_gradient_observations": actor.engine.policy.gradient_observations,
                      "actor_forward_microbatches": getattr(actor.engine.policy, "forward_microbatches", 0),
                      "actor_gradient_microbatches": getattr(actor.engine.policy, "gradient_microbatches", 0),
                      "actor_prompt_tokens": actor.engine.policy.forward_prompt_tokens,
                      "actor_gradient_prompt_tokens": getattr(actor.engine.policy, "gradient_prompt_tokens", 0),
                      "reference_prompt_tokens": adapter.frozen.prompt_tokens,
                      "reference_forwards": adapter.frozen.forwards, "reviewer_forwards": adapter.reviewer.forwards}
            previous = completed
            batch = [tasks[int(task_rng.integers(len(tasks)))] for _ in range(args.batch_size)]
            record = exact_batch_update(actor, batch, adapter)
            task_draws += len(batch)
            completed = actor.engine.updates
            if completed - previous != increment:
                raise AssertionError("Actual optimizer steps differ from the declared batch schedule")
            after = {"actor_forwards": actor.engine.policy.forward_observations,
                     "actor_gradient_observations": actor.engine.policy.gradient_observations,
                     "actor_forward_microbatches": getattr(actor.engine.policy, "forward_microbatches", 0),
                     "actor_gradient_microbatches": getattr(actor.engine.policy, "gradient_microbatches", 0),
                     "actor_prompt_tokens": actor.engine.policy.forward_prompt_tokens,
                     "actor_gradient_prompt_tokens": getattr(actor.engine.policy, "gradient_prompt_tokens", 0),
                     "reference_prompt_tokens": adapter.frozen.prompt_tokens,
                     "reference_forwards": adapter.frozen.forwards, "reviewer_forwards": adapter.reviewer.forwards}
            costs = {key: after[key] - before[key] for key in before}
            costs.update(wall_seconds=time.monotonic() - tick, **record.get("calls", {}))
            for key, value in costs.items():
                manifest["accounting"][key] = manifest["accounting"].get(key, 0) + value
                attempt["accounting"][key] = attempt["accounting"].get(key, 0) + value
            with (output / "updates.jsonl").open("a") as stream:
                stream.write(json.dumps({"attempt": attempt_index, "updates": completed, "episodes": episodes,
                                         "task_draws": task_draws, "costs": costs, **record}, allow_nan=False) + "\n")
            manifest.update(status="training", completed_updates=completed, episodes=episodes, task_draws=task_draws)
            write_json(manifest_path, manifest)
            print(json.dumps({"updates": completed, "episodes": episodes, "task_draws": task_draws,
                              "wall_seconds": costs["wall_seconds"]}), flush=True)
            if (completed in nodes if nodes is not None else completed % args.checkpoint_interval == 0 or completed == args.updates):
                calibrate(save())
        last = manifest["checkpoints"][-1]
        manifest.update(status="complete", final_adapter=str(output / last["directory"] / "actor/adapter"),
                        selection="Final fixed-budget checkpoint; no test/OOD checkpoint selection")
    except Exception as error:
        manifest.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        manifest["last_attempt_wall_seconds"] = time.monotonic() - started
        attempt.update(status=manifest["status"], wall_seconds=manifest["last_attempt_wall_seconds"],
                       finished_at_utc=datetime.now(UTC).isoformat())
        write_json(manifest_path, manifest)
        if adapter:
            adapter.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
