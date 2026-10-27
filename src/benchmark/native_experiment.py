"""Terminal-Bench Lite accuracy and HRP with one global scheduler."""
from __future__ import annotations

import argparse
import asyncio
import ipaddress
import json
import os
import statistics
import shutil
import sys
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path

from harbor.trial.trial import Trial

from src.benchmark.native_gateway import NativeGateway, command, write_json
from src.benchmark.native_runtime import (ROOT, ContainerRuntime, container_gateway,
                                         grade_tb2, source_fingerprints, start_tb2, checkpoint_fingerprint,
                                         verify_local_model_service)
from src.benchmark.terminal_datasets import load_terminal_tasks
from src.benchmark.terminal_images import check_inventory
from src.hrp.native_measurement import (MEASUREMENT_VERSION, decision_record, fingerprint,
                                        state_diagnostics, summarize_diagnostics)
from src.adapter.native_sampler import sample_native, annotate_shell_decisions, native_boundary

SAMPLING_SCHEDULE = "proportional_interleaved_v1"

def aggregate(config: dict, runs: list[dict]) -> dict:
    """Use the planned denominator, even for missing/failed infrastructure runs."""
    groups = {}
    for benchmark in config["benchmarks"]:
        planned = config["tasks"][benchmark]
        for harness in config["harnesses"]:
            for interface in config["interfaces"]:
                values = []
                for repeat in range(1, config["repeats"]+1):
                    selected = [r for r in runs if (r["benchmark"], r["harness"], r["interface"], r["repeat"]) ==
                                (benchmark, harness, interface, repeat)]
                    by_id = {r["task_id"]: r for r in selected}
                    if len(by_id) != len(selected) or set(by_id) - set(planned):
                        raise ValueError("Duplicate or unplanned evaluation rows")
                    rows = [by_id.get(task, {"task_id": task, "status": "unfinished", "resolved": False}) for task in planned]
                    resolved = sum(bool(row.get("resolved")) for row in rows)
                    values.append({"repeat": repeat, "seed": config["seed"]+repeat-1,
                        "accuracy": resolved/len(planned), "resolved": resolved, "planned": len(planned),
                        "status_counts": dict(Counter(row["status"] for row in rows)),
                        "generation_status_counts": dict(Counter(row.get("generation", {}).get("status", "not_recorded") for row in rows)),
                        "cost": {**{key: sum(row.get("cost", {}).get(key, 0) for row in rows) for key in (
                            "http_attempts", "input_tokens", "output_tokens", "usage_missing_calls",
                            "provider_cost_usd_reported", "provider_cost_missing_calls")},
                            "tasks_without_cost_ledger": sum("cost" not in row for row in rows)},
                        # Full native traces stay in each task's result.json;
                        # avoid rewriting them in every full-suite summary.
                        "tasks": [{**{key: value for key,value in row.items() if key not in ("generation", "evaluation", "verifier")},
                                   "generation_status": row.get("generation", {}).get("status", "not_recorded")}
                                  for row in rows]})
                accuracies = [value["accuracy"] for value in values]
                groups[f"{benchmark}/{harness}/{interface}"] = {
                    "runs": values, "mean_accuracy": statistics.mean(accuracies),
                    "std_accuracy": statistics.stdev(accuracies) if len(accuracies)>1 else None,
                    "all_completed": all(r["status"] != "unfinished" for v in values for r in v["tasks"])}
    return {"metric": "mean of independent single-run success rates; not pass@3", "groups": groups,
            "rollout_model_role": config.get("rollout_model_role", "target"),
            "ceiling_scope": ("empirical teacher success rate; not optimal or KL-regularized ceiling"
                              if config.get("rollout_model_role") == "teacher" else None),
            "failure_policy": "all planned tasks in denominator; timeout, infrastructure error and unfinished are separately labeled and count as unresolved"}


async def sample_task(directory: Path, gateway: NativeGateway, config: dict,
                      *, output: Path | None = None,
                      validation_state_limit: int | None = None,
                      state_paths: list[Path] | None = None) -> dict:
    output = output or directory / "hrp"
    output.mkdir(parents=True, exist_ok=True)
    diagnostics = []
    paths = sorted(state_paths) if state_paths is not None else sorted(
        (directory / "wire").glob("*/record.json"))
    native_states = {state["state_id"]: state for p in (directory / "agent/states").glob("*.json")
                     if (state := json.loads(p.read_text())).get("state_id")}
    if validation_state_limit is not None:
        paths = paths[:validation_state_limit]
    for index, path in enumerate(paths):
        wire = json.loads(path.read_text())
        state_id = wire["state_id"]
        destination = output / wire.get("attempt", path.parent.name)
        destination.mkdir(exist_ok=True)
        original_hash = fingerprint(wire["request"])
        native = native_states.get(state_id, {})
        state_hash = fingerprint({"native_state": native, "wire_record": wire})
        boundary = native_boundary(wire, native)
        common = {"state_id": state_id, "call_kind": boundary["call_kind"], "boundary": boundary,
                  "call_index": index, "source_status": wire.get("status"),
                  "discount_weight": None, "local_pair_error": None}
        environment = wire.get("environment", {})
        if wire.get("status") == "budget_exhausted" or wire.get("charged_http_attempt") is False or "messages" not in wire["request"]:
            diagnostic = {**common, "status": "state_unavailable",
                          "reason": "No actual model call or no frozen model request",
                          "environment": environment}
            diagnostics.append(diagnostic)
            write_json(destination / "diagnostics.json", diagnostic)
            continue
        if wire.get("identity", {}).get("interface") == "native" and not native:
            diagnostic = {**common, "status": "native_state_unavailable",
                          "reason": "Frozen native decoder state is missing; raw-provider fallback is not a native sample"}
            diagnostics.append(diagnostic)
            write_json(destination / "diagnostics.json", diagnostic)
            continue
        # Older Docker versions printed a deprecation notice before the image
        # digest. Preserve the original trace and verify its recorded tag/digest.
        restored_image = None
        restore = {"verified_image": None, "status": environment.get("status", "not_captured"),
                   "full_process_restore": False, "task_filesystem_restorable": False,
                   "scope": "request sampling does not execute tools or require an environment clone",
                   "excluded_mutable_mounts": environment.get("excluded_mutable_mounts", [])}
        if environment.get("status") == "filesystem_snapshot":
            try:
                actual = (await command("docker", "image", "inspect", environment["tag"], "--format", "{{.Id}}")).strip()
                if actual != environment["image"].strip().splitlines()[-1]:
                    raise ValueError("Snapshot tag no longer identifies the recorded image")
                restored_image = actual
                restore.update(verified_image=actual, status="image_verified",
                               task_filesystem_restorable=environment.get("task_filesystem_restorable", False))
            except Exception as error:
                restore.update(status="snapshot_unavailable", error=repr(error))
        write_json(destination / "environment_restore.json", restore)
        # Rehydrate twice from disk, then verify every independently generated
        # request below leaves both the source and frozen context/tools intact.
        restored = [json.loads(path.read_text())["request"] for _ in range(2)]
        if any(fingerprint(request) != original_hash for request in restored):
            raise ValueError("Frozen request restoration failed")
        # Spread both roles through the shared worker queue at their requested
        # sample ratio (128:64 -> target, target, teacher). Index 0 in each role
        # remains the independent validation branch below. Sorting only changes
        # admission order, never sample IDs, partitions or seed derivation.
        planned = sorted(((role, sample_index)
                          for role in ("target", "teacher")
                          for sample_index in range(1, config["hrp"]["n_"+role])),
                         key=lambda item: Fraction(item[1], config["hrp"]["n_"+item[0]]))

        async def one(role: str, sample_index: int):
            n = config["hrp"]["n_"+role]
            partition = "discovery" if sample_index < max(1, int(n*config["hrp"]["discovery_fraction"])) else "estimation"
            sample_id = f"{role}-{sample_index:04d}"
            sample_path = destination / f"{sample_id}.json"
            if sample_path.exists():
                previous = json.loads(sample_path.read_text())
                if (previous["source_request_fingerprint"] != original_hash or
                    previous["config_fingerprint"] != config["fingerprint"] or
                    previous.get("source_state_fingerprint") != state_hash or
                    previous.get("measurement_version") != MEASUREMENT_VERSION):
                    raise ValueError("Sampling cache belongs to a different frozen state/configuration")
                return previous
            seed = int(fingerprint([config["seed"], config["fingerprint"], state_id, role, sample_index])[:8], 16) % (2**31)
            try:
                if wire["identity"]["interface"] == "native":
                    sampled = await sample_native(gateway=gateway, record=wire,
                        state_path=directory / "agent/states" / f"{state_id}.json",
                        output=destination / sample_id, model_role=role, seed=seed,
                        node=Path(config["runtime"]["node"]), sampling_budget=config["hrp"].get("sampling_budget"))
                else:
                    sampled = await gateway.sample(wire, role, seed=seed)
                    write_json(destination / sample_id / "transport/cost.json", sampled["cost"])
                response = annotate_shell_decisions(sampled["response"])
                for key in ("messages", "tools", "tool_choice"):
                    if sampled["request"].get(key) != wire["request"].get(key):
                        raise ValueError(f"Sampling changed native {key}")
            except Exception as error:
                sampled = {"error": repr(error), "elapsed_sec": None}
                response = {"error": {"message": repr(error)}}
            decision = decision_record(response, state_id=state_id, harness=wire["identity"]["harness"],
                sample_id=sample_id, environment_version=(environment["image"].strip().splitlines()[-1]
                    if environment.get("status") == "filesystem_snapshot" else None),
                tool_definitions=response.get("decision_tool_definitions", wire["request"].get("tools")),
                call_kind=boundary["call_kind"], boundary=boundary)
            decision["environment_available"] = restore["status"] == "image_verified"
            row = {"role": role, "partition": partition, "seed": seed,
                   "measurement_version": MEASUREMENT_VERSION,
                   "source_state_fingerprint": state_hash,
                   "source_request_fingerprint": original_hash, "config_fingerprint": config["fingerprint"],
                   "sample": sampled, "decision": decision}
            write_json(sample_path, row)
            return row

        # The first independent branches validate live request restoration before
        # releasing the rest of the per-state samples. They are discovery samples.
        first = await one("target", 0)
        second = await one("teacher", 0)
        unchanged = fingerprint(json.loads(path.read_text())["request"]) == original_hash
        same_context = all(not row["sample"].get("error") and all(
            row["sample"].get("request", {}).get(key) == wire["request"].get(key)
            for key in ("messages", "tools", "tool_choice")) for row in (first, second))
        validation = {"source_unchanged": unchanged, "same_context_and_tools": same_context,
                      "tools_executed": 0, "restores_from_disk": 2,
                      "scope": "frozen next model request; no full controller/process restore",
                      "target_http_status": first["sample"].get("http_status"),
                      "teacher_http_status": second["sample"].get("http_status"),
                      "cross_model_private_replay_unverified": any(row["sample"].get("cross_model_private_replay_unverified") for row in (first,second))}
        write_json(destination / "replay_validation.json", validation)
        if not unchanged:
            raise ValueError("Sampling contaminated the trajectory state")
        if not same_context or any(row["sample"].get("http_status") != 200 for row in (first, second)):
            diagnostic = {**common, "status": "sampling_interface_validation_failed", "validation": validation}
            diagnostics.append(diagnostic)
            write_json(destination / "diagnostics.json", diagnostic)
            continue
        rows = [first, second] + await asyncio.gather(*(one(role, i) for role, i in planned))
        diagnostic = state_diagnostics(rows, state_id=state_id, tau=config["hrp"]["tau"], alpha=config["hrp"]["alpha"],
                                       min_pair_count=config["hrp"]["min_pair_count"])
        if any(row["sample"].get("cross_model_private_replay_unverified") for row in rows):
            diagnostic["status"] = "private_replay_comparability_unverified"
            diagnostic["supported_pair_subset_mean"] = None
            for layer in diagnostic["layers"].values():
                layer["supported_pair_subset_mean"] = None
                layer["comparison_valid"] = False
                for pair in layer["pairs"]:
                    pair["squared_difference"] = None
            diagnostic["limitations"].append("Identical wire fields do not establish cross-model interpretation of private reasoning replay.")
        diagnostic.update({key: value for key, value in common.items() if key != "local_pair_error"})
        diagnostic["environment_restore"] = restore
        diagnostic["sampling_scope"] = "frozen native next-request distribution; execution equivalence unverified"
        diagnostics.append(diagnostic)
        write_json(destination / "diagnostics.json", diagnostic)
    represented = {state["state_id"] for state in diagnostics}
    for state_path in sorted((directory / "agent/states").glob("*.json")) \
            if validation_state_limit is None and state_paths is None else []:
        native = json.loads(state_path.read_text())
        if native.get("state_id") not in represented:
            diagnostics.append({"state_id": native.get("state_id"), "status": "no_recorded_http_attempt",
                                "call_kind": native.get("call_kind", "unclassified"),
                                "boundary": native_boundary({"state_id": native.get("state_id")}, native),
                                "local_pair_error": None, "native_state_path": str(state_path),
                                "reason": "Native boundary retained; no corresponding transport state to sample"})
    costs = {}
    for role in ("target", "teacher"):
        ledgers = [json.loads(path.read_text()) for path in output.glob(f"call-*/{role}-*/transport/cost.json")]
        costs[role] = {key: sum(ledger.get(key, 0) for ledger in ledgers) for key in (
            "http_attempts", "input_tokens", "output_tokens", "usage_missing_calls",
            "provider_cost_usd_reported", "provider_cost_missing_calls")}
        costs[role]["sample_ledgers"] = len(ledgers)
    nodes = [state["boundary"] for state in diagnostics if state.get("boundary")]
    edges = [{"source_logical_call_id": node[key], "target_call_id": node.get("call_id"),
              "target_logical_call_id": node.get("logical_call_id"), "relation": relation}
             for node in nodes for key, relation in (("parent_state_id", "parent_call"), ("previous_state_id", "native_predecessor"))
             if node.get(key)]
    write_json(output / "call_graph.json", {"nodes": nodes, "edges": edges,
        "scope": "recorded native lineage and overlapping HTTP attempts; no imposed serial execution",
        "child_boundaries_without_parent_call": sum(node.get("agent_role") == "subagent" and not node.get("parent_state_id") for node in nodes)})
    summary = {"measurement_version": MEASUREMENT_VERSION,
               "states": diagnostics,
               "all_recorded_boundaries_retained": validation_state_limit is None and state_paths is None,
               "validation_only": validation_state_limit is not None,
               "diagnostics": summarize_diagnostics(diagnostics),
               "by_call_kind": {kind: summarize_diagnostics([s for s in diagnostics if s.get("call_kind", "unclassified") == kind])
                                for kind in sorted({s.get("call_kind", "unclassified") for s in diagnostics})},
               "logical_model_boundaries": len({s["state_id"] for s in diagnostics}),
               "trajectory_cost": json.loads((directory / "cost.json").read_text()) if (directory / "cost.json").exists() else None,
               "sampling_cost": costs,
               "full_space_energy": None, "predicted_success": None, "harness_ranking": None,
               "discount_convention": "no global discount assigned from arrival order; gamma retained for future calibrated causal weighting",
               "calibration": "not fitted; teacher performance and independent held-out outcome validation required"}
    write_json(output / "summary.json", summary)
    return summary


async def run_groups(config: dict, run_group) -> None:
    """Build groups in the requested scheduler order and enforce its outer policy."""
    groups = []
    slot = 0
    if config.get("scheduling", "parallel") == "sequential":
        identities = ((benchmark, harness, interface, repeat)
            for benchmark in config["benchmarks"] for harness in config["harnesses"]
            for interface in config["interfaces"] for repeat in range(1, config["repeats"]+1))
    else:
        # Match the original concurrent launcher: different harnesses and both
        # benchmarks enter the outer semaphore before later repeats.
        identities = ((benchmark, harness, interface, repeat)
            for interface in config["interfaces"] for repeat in range(1, config["repeats"]+1)
            for harness in config["harnesses"] for benchmark in config["benchmarks"])
    for benchmark, harness, interface, repeat in identities:
        assigned = [(task_id, slot+i) for i, task_id in enumerate(config["tasks"][benchmark])]
        slot += len(assigned)
        groups.append((benchmark, harness, interface, repeat, assigned))
    if config.get("scheduling", "parallel") == "sequential":
        for group in groups:
            await run_group(*group)
    else:
        await asyncio.gather(*(run_group(*group) for group in groups))


async def execute(config: dict, output: Path) -> None:
    rollout_role = config.get("rollout_model_role", "target")
    if rollout_role == "teacher" and config["mode"] == "hrp":
        raise ValueError("Teacher rollouts use accuracy/trajectory mode; HRP occupancy remains target-only")
    config["hrp"]["measurement_version"] = MEASUREMENT_VERSION
    config["hrp"]["sampling_schedule"] = SAMPLING_SCHEDULE
    config["concurrency"].setdefault("native_sample_workers", 40)
    images = await check_inventory(config, config["benchmarks"])
    cases = {b: load_terminal_tasks(config, b) for b in config["benchmarks"]}
    config["tasks"] = {b: list(selected) for b, selected in cases.items()}
    # Resolve before hashing: the manifest records the actual task deadlines,
    # not just an unresolved null or a single suite-wide timeout.
    config["task_budgets"] = {b: {task_id: {
        "agent": {**config["budget"], "wall_time_sec": (
            case["agent_timeout_sec"] if config.get("timeout_policy") == "benchmark_default"
            else config["budget"]["wall_time_sec"])},
        "agent_setup_timeout_sec": Trial._AGENT_SETUP_TIMEOUT_SEC,
        "verifier_timeout_sec": case["verifier_timeout_sec"],
    } for task_id, case in selected.items()} for b, selected in cases.items()}
    config["prepared_images"] = images
    runtime = ContainerRuntime(Path(config["runtime"]["node"]), Path(config["runtime"]["python"]))
    target = {**config["target"], "api_key": os.environ.get("HRP_TARGET_API_KEY", "EMPTY")}
    teacher = {**config["teacher"], "api_key": os.environ.get("OPENROUTER_API_KEY", "")}
    if (config["mode"] == "hrp" or rollout_role == "teacher") and not teacher["api_key"]:
        # The existing credentials file stays on the host and is never mounted.
        credentials = json.loads((ROOT / "api.json").read_text()) if (ROOT / "api.json").exists() else {}
        teacher["api_key"] = credentials.get("openrouter_api_key", credentials.get("api_key", credentials.get("key", "")))
        if not teacher["api_key"]:
            raise ValueError("Set OPENROUTER_API_KEY for teacher requests")
    limits = config["concurrency"]
    provider_retry = config.get("provider_retry", {})
    gateway = NativeGateway(target, teacher, local_concurrency=limits["local_requests"],
                            teacher_concurrency=limits["teacher_requests"],
                            native_worker_concurrency=limits["native_sample_workers"],
                            openrouter_max_retries=provider_retry.get("openrouter_max_retries", 10),
                            openrouter_retry_initial_sec=provider_retry.get("initial_delay_sec", 1.),
                            openrouter_retry_max_sec=provider_retry.get("max_delay_sec", 30.))
    containers = asyncio.Semaphore(limits["containers"])
    evaluators = asyncio.Semaphore(limits["evaluators"])
    outer = asyncio.Semaphore(limits["outer_runs"])
    await gateway.start()
    telemetry_stop = asyncio.Event()
    telemetry_task = None
    all_results: list[dict] = []
    try:
        if rollout_role == "target":
            config["model_service_verification"] = verify_local_model_service(config)
            async with gateway.client.get(config["target"]["api_base"].rstrip("/")+"/models") as response:
                response.raise_for_status()
                models = (await response.json())["data"]
            served = next((model for model in models if model["id"] == config["target"]["id"]), None)
            if served is None or served.get("max_model_len") != config["target"]["context_window"]:
                raise ValueError("Target service model/context does not match the experiment manifest")
            if served.get("root") != config["target"]["checkpoint"]:
                raise ValueError("Target service is not serving the pinned local checkpoint path")
            config["served_model"] = {key: served.get(key) for key in ("id", "root", "max_model_len", "owned_by")}
            config["target_checkpoint_files"] = await asyncio.to_thread(checkpoint_fingerprint, Path(config["target"]["checkpoint"]))
        else:
            config["model_service_verification"] = {"status": "not_used", "reason": "teacher-only remote rollouts"}
        config["sources"] = source_fingerprints()
        config["fingerprint"] = fingerprint(config)
        experiment = output / config["fingerprint"][:20]
        experiment.mkdir(parents=True, exist_ok=True)
        manifest = experiment / "manifest.json"
        if manifest.exists() and json.loads(manifest.read_text()) != config:
            raise ValueError("Experiment fingerprint collision or incompatible manifest")
        write_json(manifest, config)
        source_snapshot = experiment / "implementation/src"
        if not source_snapshot.exists():
            shutil.copytree(ROOT / "src", source_snapshot, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        runtime.source_snapshot = source_snapshot
        write_json(output / "latest.json", {"experiment": str(experiment), "fingerprint": config["fingerprint"]})
        evaluation_started = time.time()
        async def telemetry():
            last_progress = 0.
            while not telemetry_stop.is_set():
                row = {"time": time.time(), "active_model_requests": dict(gateway.active),
                       "peak_model_requests": dict(gateway.peak)}
                try:
                    gpu = await command("nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                                        "--format=csv,noheader,nounits", timeout=10)
                    row["gpus"] = [{"index": int(parts[0]), "memory_mib": int(parts[1]), "utilization_percent": int(parts[2])}
                                   for line in gpu.splitlines() if (parts := line.split(","))]
                    memory = dict(line.split(":",1) for line in Path("/proc/meminfo").read_text().splitlines())
                    row["memory_available_kib"] = int(memory["MemAvailable"].split()[0])
                except Exception as error:
                    row["telemetry_error"] = repr(error)
                with (experiment / "resources.jsonl").open("a") as stream:
                    stream.write(json.dumps(row)+"\n")
                if time.time()-last_progress >= config.get("progress_interval_sec", 3600):
                    progress = {"time": time.time(), "elapsed_sec": time.time()-evaluation_started,
                        "completed": len(all_results), "benchmarks": {benchmark: {
                            "planned": len(config["tasks"][benchmark])*len(config["harnesses"])*len(config["interfaces"])*config["repeats"],
                            "completed": sum(result["benchmark"] == benchmark for result in all_results),
                            "resolved": sum(result["benchmark"] == benchmark and result.get("resolved", False) for result in all_results)}
                            for benchmark in config["benchmarks"]},
                        "status_counts": dict(Counter(result["status"] for result in all_results)),
                        "active_model_requests": dict(gateway.active)}
                    write_json(experiment / "progress.json", progress)
                    with (experiment / "hourly_progress.jsonl").open("a") as stream:
                        stream.write(json.dumps(progress)+"\n")
                    print(json.dumps({"event": "hourly_progress", **progress}), flush=True)
                    last_progress = time.time()
                try:
                    await asyncio.wait_for(telemetry_stop.wait(), 15)
                except TimeoutError:
                    pass
        telemetry_task = asyncio.create_task(telemetry())

        async def task_run(benchmark, harness, interface, repeat, task_id, index):
            identity = {"benchmark": benchmark, "harness": harness, "interface": interface,
                        "repeat": repeat, "task_id": task_id, "config_fingerprint": config["fingerprint"]}
            directory = experiment / benchmark / harness / interface / f"repeat-{repeat}" / task_id
            directory.mkdir(parents=True, exist_ok=True)
            result_path = directory / "result.json"
            if result_path.exists():
                result = json.loads(result_path.read_text())
                if any(result.get(k) != v for k,v in identity.items()):
                    raise ValueError("Evaluation cache identity mismatch")
                all_results.append(result)
                if config["mode"] == "hrp":
                    await sample_task(directory, gateway, config)
                return
            agent_output = directory / "agent"
            agent_output.mkdir(exist_ok=True)
            environment = None
            container = None
            token = None
            started = time.time()
            result = {**identity, "status": "unfinished", "resolved": False}
            async with containers:
                try:
                    task, environment, container, cwd = await start_tb2(
                        Path(cases[benchmark][task_id]["path"]), directory, runtime, index,
                        network_cidr=config["network"]["terminal_cidr"], prepared_image=images[benchmark][task_id],
                        persistent_env=config["network"].get("persistent_env", {}))
                    instruction = task.instruction
                    task_budget = config["task_budgets"][benchmark][task_id]
                    result["budgets"] = task_budget
                    token, session = gateway.register(directory, identity=identity, budget=task_budget["agent"],
                         snapshots=config["mode"] in ("hrp", "trajectory") and config["hrp"].get("capture_filesystem_snapshots", False),
                         container_id=container, model_role=rollout_role)
                    host = await container_gateway(container)
                    model = {**config[rollout_role], "api_base": f"http://{host}:{gateway.port}/s/{token}/v1", "api_key": "EMPTY"}
                    agent_config = {"operation": "run", "harness": harness, "interface": interface,
                                    "instruction": instruction, "cwd": cwd, "model": model,
                                    "budget": task_budget["agent"], "seed": config["seed"]+repeat-1,
                                    "setup_timeout_sec": task_budget["agent_setup_timeout_sec"],
                                    "user": task.config.agent.user,
                                    "profile": config.get("harness_profiles", {}).get(harness, {})}
                    if config.get("forced_first_action") is not None:
                        agent_config["forced_first_action"] = config["forced_first_action"]
                    def on_run_start():
                        # Harbor budgets agent execution separately from setup.
                        session["started"] = time.time()
                    if harness == "terminus-2":
                        from src.adapter.terminus_native import run_terminus
                        generation = await run_terminus(environment, agent_config, agent_output, on_run_start=on_run_start)
                        if generation["status"] == "generation_timeout":
                            # A stopped HTTP/controller task does not stop a
                            # foreground program left running in tmux.
                            await command("docker", "stop", "--time", "2", container)
                            await command("docker", "start", container)
                    else:
                        generation = await runtime.launch(container, agent_config, agent_output, on_run_start=on_run_start)
                    result["generation"] = generation
                    async with evaluators:
                        result.update(await grade_tb2(task, environment, directory))
                except Exception as error:
                    result.update(status="infrastructure_error", error=repr(error))
                finally:
                    result["elapsed_sec"] = time.time()-started
                    if (directory / "cost.json").exists():
                        result["cost"] = json.loads((directory / "cost.json").read_text())
                    write_json(result_path, result)
                    all_results.append(result)
                    if token:
                        gateway.sessions.pop(token, None)
                    if environment:
                        await environment.stop(delete=True)
                    elif container:
                        await command("docker", "rm", "-f", container)
            if config["mode"] == "hrp" and result.get("generation"):
                await sample_task(directory, gateway, config)
            write_json(experiment / "summary.json", aggregate(config, all_results))
            print(json.dumps({**identity, "status": result["status"], "resolved": result["resolved"]}), flush=True)

        async def run_group(benchmark, harness, interface, repeat, assigned):
            async with outer:
                print(json.dumps({"event": "group_start", "benchmark": benchmark, "harness": harness,
                    "interface": interface, "repeat": repeat, "tasks_per_run": limits["tasks_per_run"]}), flush=True)
                inner = asyncio.Semaphore(limits["tasks_per_run"])
                async def wrapped(task_id, index):
                    async with inner:
                        await task_run(benchmark, harness, interface, repeat, task_id, index)
                await asyncio.gather(*(wrapped(task_id, index) for task_id,index in assigned))
                print(json.dumps({"event": "group_finished", "benchmark": benchmark, "harness": harness,
                    "interface": interface, "repeat": repeat}), flush=True)
        write_json(experiment / "summary.json", aggregate(config, []))
        await run_groups(config, run_group)
        write_json(experiment / "summary.json", aggregate(config, all_results))
        write_json(experiment / "resources.json", {"peak_model_requests": gateway.peak, "limits": limits,
                   "capacity_claim": "only the observed run shape; no extrapolation from short trajectories"})
    finally:
        telemetry_stop.set()
        if telemetry_task:
            await telemetry_task
        await gateway.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "src/benchmark/config/native_experiment.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("accuracy", "trajectory", "hrp"))
    parser.add_argument("--rollout-model-role", choices=("target", "teacher"),
                        help="Teacher runs independent native trajectories for empirical ceiling measurement")
    parser.add_argument("--benchmarks", nargs="+", choices=("tb_lite",))
    parser.add_argument("--interfaces", nargs="+", choices=("native",))
    parser.add_argument("--harnesses", nargs="+", choices=("pi", "deepagents", "deepseek", "terminus-2", "mini-swe-agent"))
    parser.add_argument("--cases", type=int)
    parser.add_argument("--task-ids", nargs="+", help="Run only these task IDs; requires exactly one benchmark")
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--scheduling", choices=("sequential", "parallel"))
    parser.add_argument("--tasks-per-run", type=int)
    parser.add_argument("--max-steps", type=int, help="Top-level step cap; 0 disables it (bundled config: 50)")
    parser.add_argument("--terminal-cidr", help="Private IPv4 /16 used to allocate isolated /28 task networks")
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--n-target", type=int, help="HRP target samples per recorded model-call boundary")
    parser.add_argument("--n-teacher", type=int, help="HRP teacher samples per recorded model-call boundary")
    parser.add_argument("--teacher-reasoning-effort", choices=("low", "medium", "high"))
    parser.add_argument("--target-api-base")
    parser.add_argument("--gpu-ids", nargs="+", type=int)
    parser.add_argument("--local-requests", type=int)
    parser.add_argument("--teacher-requests", type=int)
    parser.add_argument("--native-sample-workers", type=int)
    parser.add_argument("--evaluators", type=int)
    parser.add_argument("--check-images", action="store_true", help="Validate pinned tasks/images without starting model calls")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    config["target"]["checkpoint"] = str(Path(os.environ.get(
        "HRP_MODEL_DIR", config["target"]["checkpoint"])).expanduser().resolve())
    config["runtime"]["node"] = shutil.which(config["runtime"]["node"]) or config["runtime"]["node"]
    config["runtime"]["python"] = sys.executable if config["runtime"]["python"] == "python" else config["runtime"]["python"]
    for key in ("mode", "benchmarks", "interfaces", "harnesses", "repeats", "scheduling", "rollout_model_role"):
        if getattr(args,key) is not None:
            config[key] = getattr(args,key)
    if args.cases is not None:
        if args.cases <= 0:
            parser.error("--cases must be positive")
        config["case_limit"] = args.cases
    if args.task_ids is not None:
        if len(config["benchmarks"]) != 1:
            parser.error("--task-ids requires exactly one selected benchmark")
        if args.cases is not None:
            parser.error("--task-ids cannot be combined with --cases")
        if len(args.task_ids) != len(set(args.task_ids)):
            parser.error("--task-ids must be unique")
        config.pop("case_limit", None)
        config["task_ids"] = {config["benchmarks"][0]: args.task_ids}
    if args.seed is not None:
        config["seed"] = args.seed
    if args.max_steps is not None:
        if args.max_steps < 0:
            parser.error("--max-steps must be non-negative; 0 means unlimited")
        config["budget"]["max_steps"] = args.max_steps or None
    if args.terminal_cidr is not None:
        try:
            network = ipaddress.ip_network(args.terminal_cidr)
        except ValueError as error:
            parser.error(f"--terminal-cidr is invalid: {error}")
        if network.version != 4 or network.prefixlen != 16 or not network.is_private:
            parser.error("--terminal-cidr must be a private IPv4 /16")
        config["network"]["terminal_cidr"] = str(network)
    if args.max_tokens:
        config["target"]["max_tokens"] = args.max_tokens
    for role in ("target", "teacher"):
        value = getattr(args, "n_"+role)
        if value is not None:
            if value <= 0:
                parser.error(f"--n-{role} must be positive")
            config["hrp"]["n_"+role] = value
    if args.teacher_reasoning_effort is not None:
        config["teacher"]["reasoning_effort"] = args.teacher_reasoning_effort
    if args.target_api_base:
        config["target"]["api_base"] = args.target_api_base
    if args.gpu_ids:
        if len(args.gpu_ids) != len(set(args.gpu_ids)) or min(args.gpu_ids) < 0:
            parser.error("--gpu-ids must contain distinct non-negative device IDs")
        config["runtime"]["gpu_ids"] = args.gpu_ids
        config["runtime"]["data_parallel_size"] = len(args.gpu_ids)
    for argument in ("local_requests", "teacher_requests", "native_sample_workers", "evaluators", "tasks_per_run"):
        value = getattr(args, argument)
        if value is not None:
            if value <= 0:
                parser.error(f"--{argument.replace('_', '-')} must be positive")
            config["concurrency"][argument] = value
    if args.check_images:
        ready = asyncio.run(check_inventory(config, config["benchmarks"]))
        print(json.dumps({"images_ready": {b: len(rows) for b,rows in ready.items()}}))
        return
    config["tasks"] = {}
    args.output.mkdir(parents=True, exist_ok=True)
    asyncio.run(execute(config, args.output.resolve()))


if __name__ == "__main__":
    main()
