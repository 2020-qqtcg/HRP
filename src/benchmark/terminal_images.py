"""Prepare pinned task images once; refuse incomplete inventories at evaluation time."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path

from src.benchmark.native_gateway import command, write_json
from src.benchmark.terminal_datasets import ROOT, load_terminal_tasks, checkout_terminal_datasets
from src.hrp.native_measurement import fingerprint


async def inspect_image(ref: str) -> dict | None:
    try:
        info = json.loads(await command("docker", "image", "inspect", ref))[0]
    except RuntimeError:
        return None
    if info.get("Architecture") != "amd64" or info.get("Os") != "linux":
        raise ValueError(f"Expected linux/amd64 image: {ref}")
    return info


async def check_inventory(config: dict, benchmarks: list[str]) -> dict:
    path = ROOT / config["image_preparation"]["manifest"]
    if not path.exists():
        raise ValueError(f"Missing image inventory: run scripts/prepare_terminal_images.sh ({path})")
    inventory = json.loads(path.read_text())
    selected = {}
    for benchmark in benchmarks:
        selected[benchmark] = {}
        for task_id, task in load_terminal_tasks(config, benchmark).items():
            row = inventory.get("tasks", {}).get(benchmark, {}).get(task_id)
            if not row or row.get("status") != "ready" or row.get("task_fingerprint") != task["task_fingerprint"]:
                raise ValueError(f"Image not prepared for pinned task {benchmark}/{task_id}")
            info = await inspect_image(row["local_ref"])
            if info is None or info["Id"] != row["image_id"]:
                raise ValueError(f"Prepared image missing or changed: {benchmark}/{task_id}")
            selected[benchmark][task_id] = row
    return selected


async def prepare(config: dict, benchmarks: list[str]) -> None:
    await asyncio.to_thread(checkout_terminal_datasets, config, benchmarks)
    settings = config["image_preparation"]
    destination = ROOT / settings["manifest"]
    cache = Path(settings["oci_cache"]).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    logs = destination.parent / "image_logs"
    logs.mkdir(parents=True, exist_ok=True)
    inventory = json.loads(destination.read_text()) if destination.exists() else {"schema": "terminal-images-v1", "tasks": {}}
    inventory.update(datasets={b: config["datasets"][b] for b in benchmarks},
                     preparation={"regctl": settings["regctl"], "oci_cache": str(cache),
                         "download_workers": 1, "build_workers": settings["build_workers"],
                         "network": "host during build; task runtime network remains benchmark controlled",
                         "proxy_variables_present": [k for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY") if os.environ.get(k)]})
    tasks = {b: load_terminal_tasks(config, b) for b in benchmarks}
    download_lock = asyncio.Lock()
    downloaded = {}

    async def fetch(ref: str) -> dict:
        async with download_lock:
            if ref in downloaded:
                return downloaded[ref]
            existing = await inspect_image(ref)
            if existing is None:
                regctl = str(Path(settings["regctl"]).expanduser())
                oci = f"ocidir://{cache}:image-{fingerprint(ref)[:24]}"
                log_path = logs / f"pull-{fingerprint(ref)[:24]}.log"
                for attempt in range(settings["download_attempts"]):
                    try:
                        with log_path.open("a") as log:
                            process = await asyncio.create_subprocess_exec(regctl, "image", "copy", "--platform", "linux/amd64", ref, oci,
                                stdout=log, stderr=log)
                            code = await asyncio.wait_for(process.wait(), settings["download_timeout_sec"])
                        if code:
                            raise RuntimeError(f"regctl copy exited {code}; see {log_path}")
                        with tempfile.TemporaryDirectory(prefix="hrp-terminal-image-", dir=cache.parent) as tmp:
                            archive = Path(tmp) / "image.tar"
                            await command(regctl, "image", "export", oci, str(archive), "--name", ref, timeout=1800)
                            await command("docker", "load", "-i", str(archive), timeout=1800)
                        break
                    except Exception as error:
                        if 'process' in locals() and process.returncode is None:
                            process.kill()
                            await process.wait()
                        print(json.dumps({"event": "image_download_retry", "image": ref,
                            "attempt": attempt+1, "error": repr(error)}), flush=True)
                        if attempt+1 == settings["download_attempts"]:
                            raise
                        await asyncio.sleep(min(60, 5*2**attempt))
                existing = await inspect_image(ref)
                if existing is None:
                    raise RuntimeError(f"Image import did not create {ref}")
            downloaded[ref] = existing
            return existing

    build_limit = asyncio.Semaphore(settings["build_workers"])

    async def one(benchmark: str, task_id: str, task: dict) -> None:
        rows = inventory["tasks"].setdefault(benchmark, {})
        previous = rows.get(task_id)
        if previous and previous.get("status") == "ready" and previous["task_fingerprint"] == task["task_fingerprint"]:
            info = await inspect_image(previous["local_ref"])
            if info and info["Id"] == previous["image_id"]:
                return
        row = {**task, "status": "preparing"}
        rows[task_id] = row
        write_json(destination, inventory)
        try:
            if task["source_image"]:
                info = await fetch(task["source_image"])
                row["origin"] = "upstream_prebuilt"
            else:
                async with build_limit:
                    environment = Path(task["path"]) / "environment"
                    dockerfile = (environment / "Dockerfile").read_text()
                    bases = {}
                    stages = set()
                    for line in dockerfile.splitlines():
                        match = re.match(r"(?i)^FROM\s+(\S+)(?:\s+AS\s+(\S+))?\s*$", line)
                        if not match:
                            continue
                        base, stage = match.groups()
                        if base.lower() not in stages and base != "scratch":
                            base_info = await fetch(base)
                            pinned = "hrp-terminal-base:"+base_info["Id"].split(":")[1][:24]
                            await command("docker", "tag", base, pinned)
                            bases[base] = {"image_id": base_info["Id"], "local_ref": pinned}
                        if stage:
                            stages.add(stage.lower())
                    locked = re.sub(r"(?im)^(FROM\s+)(\S+)", lambda m: m[1]+bases.get(m[2], {}).get("local_ref", m[2]), dockerfile)
                    tag = f"hrp-terminal-build/{benchmark}-{task_id}:{task['task_fingerprint'][:20]}"
                    log_path = logs / f"build-{benchmark}-{task_id}.log"
                    with tempfile.TemporaryDirectory(prefix="hrp-terminal-build-") as tmp:
                        recipe = Path(tmp) / "Dockerfile"
                        recipe.write_text(locked)
                        args = ["docker", "build", "--network", "host", "--pull=false", "-f", str(recipe), "-t", tag]
                        for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "all_proxy", "no_proxy"):
                            if os.environ.get(name):
                                args += ["--build-arg", name]
                        args.append(str(environment))
                        with log_path.open("w") as log:
                            process = await asyncio.create_subprocess_exec(*args, stdout=log, stderr=log)
                            try:
                                code = await asyncio.wait_for(process.wait(), settings["build_timeout_sec"])
                            except BaseException:
                                process.kill()
                                await process.wait()
                                raise
                        if code:
                            raise RuntimeError(f"Docker build exited {code}; see {log_path}")
                    info = await inspect_image(tag)
                    row.update(origin="upstream_dockerfile", base_images=bases,
                               original_dockerfile_sha256=hashlib.sha256(dockerfile.encode()).hexdigest(),
                               locked_dockerfile_sha256=hashlib.sha256(locked.encode()).hexdigest(),
                               recipe_change="FROM aliases point to identical locally pinned base images")
            local_ref = f"hrp-terminal/{benchmark}-{task_id}:"+info["Id"].split(":")[1][:24]
            await command("docker", "tag", info["Id"], local_ref)
            row.update(status="ready", local_ref=local_ref, image_id=info["Id"],
                repo_digests=info.get("RepoDigests", []), rootfs_layers=info["RootFS"]["Layers"],
                architecture=info["Architecture"], os=info["Os"], size_bytes=info["Size"])
        except Exception as error:
            row.update(status="error", error=repr(error))
        row["finished_at"] = time.time()
        write_json(destination, inventory)
        print(json.dumps({"benchmark": benchmark, "task": task_id, "status": row["status"],
                          "error": row.get("error")}), flush=True)

    await asyncio.gather(*(one(b, t, item) for b, selected in tasks.items() for t,item in selected.items()))
    await check_inventory(config, benchmarks)
    print(json.dumps({"event": "all_images_ready", "tasks": sum(map(len, tasks.values())), "manifest": str(destination)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "src/benchmark/config/native_experiment.json")
    parser.add_argument("--benchmarks", nargs="+", choices=("tb_lite",))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    benchmarks = args.benchmarks or config["benchmarks"]
    if args.check:
        inventory = asyncio.run(check_inventory(config, benchmarks))
        print(json.dumps({"images_ready": {b: len(rows) for b,rows in inventory.items()}}))
    else:
        asyncio.run(prepare(config, benchmarks))


if __name__ == "__main__":
    main()
