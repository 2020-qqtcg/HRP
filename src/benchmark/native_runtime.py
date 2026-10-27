"""Task-container runtime for native harnesses and the retained Bash control."""
from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import os
import subprocess
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urlparse
import harbor

from harbor.environments.docker.docker import DockerEnvironment
from harbor.models.task.task import Task
from harbor.models.trial.paths import TrialPaths
from harbor.trial.trial import Trial
from harbor.verifier.verifier import Verifier

from src.benchmark.native_gateway import command, write_json

ROOT = Path(__file__).resolve().parents[2]


def source_fingerprints(*, measurement_root: Path | None = None) -> dict:
    """Record actual provenance and fingerprint source AND runtime files."""
    result = {}
    for name, folder in (("pi", ROOT / "agents/pi"), ("deepseek", ROOT / "agents/deepseek-harness"),
                         ("deepagents", ROOT / "agents/deepagents/libs/deepagents"),
                         ("mini-swe-agent", ROOT / "agents/mini-swe-agent"),
                         ("terminus-2", Path(harbor.__file__).parent), ("measurement", measurement_root or ROOT / "src")):
        digest, count = hashlib.sha256(), 0
        for current, dirs, files in os.walk(folder):
            dirs[:] = sorted(d for d in dirs if d not in {"node_modules", ".git", "__pycache__", ".pytest_cache"})
            for filename in sorted(files):
                path = Path(current) / filename
                if path.suffix not in {".py", ".ts", ".js", ".mjs", ".json", ".yml", ".yaml", ".toml", ".lock"}:
                    continue
                digest.update(str(path.relative_to(folder)).encode() + b"\0" + path.read_bytes() + b"\0")
                count += 1
        result[name] = {"sha256": digest.hexdigest(), "files": count, "upstream_git_commit": None,
                        "scope": "source and built runtime; node_modules excluded, lockfiles included"}
        if (folder / ".git").exists():
            result[name]["upstream_git_commit"] = subprocess.check_output(
                ["git", "-C", str(folder), "rev-parse", "HEAD"], text=True).strip()
            result[name]["upstream_worktree_changes"] = subprocess.check_output(
                ["git", "-C", str(folder), "status", "--porcelain"], text=True).splitlines()
    result["terminus-2"]["harbor_version"] = version("harbor")
    result["protocol_dependencies"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (ROOT / "uv.lock", ROOT / "pyproject.toml", ROOT / "scripts/eval_native_harnesses.sh",
                     ROOT / "scripts/hrp_native_harnesses.sh",
                      ROOT / "scripts/prepare_terminal_images.sh")}
    return result


def checkpoint_fingerprint(folder: Path) -> dict:
    files = sorted([*folder.glob("*.safetensors"), *folder.glob("*.json")])
    if not files or not any(path.suffix == ".safetensors" for path in files):
        raise ValueError(f"Expected a local safetensors checkpoint at {folder}")
    result = {}
    for path in files:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(8*1024*1024):
                digest.update(chunk)
        result[path.name] = {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}
    return result


def verify_local_model_service(config: dict) -> dict:
    """Check actual local launch flags; never silently reuse a different server."""
    target, runtime = config["target"], config["runtime"]
    endpoint = urlparse(target["api_base"])
    if endpoint.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("This profile requires an auditable local vLLM service")
    candidates = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv = (proc / "cmdline").read_bytes().decode().rstrip("\0").split("\0")
            if "vllm.entrypoints.openai.api_server" not in argv:
                continue
            flags = {item: argv[index+1] if index+1 < len(argv) and not argv[index+1].startswith("--") else True
                     for index,item in enumerate(argv) if item.startswith("--")}
            if str(flags.get("--port", 8000)) != str(endpoint.port or 80):
                continue
            env = dict(item.split("=",1) for item in (proc / "environ").read_bytes().decode().split("\0") if "=" in item)
            candidates.append((flags, env.get("CUDA_VISIBLE_DEVICES")))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    if len(candidates) != 1:
        raise ValueError("Cannot identify the local model service; start it with eval_native_harnesses.sh")
    flags, devices = candidates[0]
    expected = {"--model": target["checkpoint"], "--served-model-name": target["id"],
                "--max-model-len": target["context_window"],
                "--tensor-parallel-size": runtime["tensor_parallel_size"],
                "--data-parallel-size": runtime["data_parallel_size"], "--dtype": runtime["dtype"],
                "--gpu-memory-utilization": runtime["gpu_memory_utilization"],
                "--max-num-seqs": runtime["max_num_seqs_per_rank"],
                "--max-num-batched-tokens": runtime["max_num_batched_tokens"],
                "--watermark": runtime["watermark"], "--reasoning-parser": runtime["reasoning_parser"],
                "--tool-call-parser": runtime["tool_call_parser"],
                "--enable-prefix-caching": runtime["prefix_caching"],
                "--enable-chunked-prefill": runtime["chunked_prefill"],
                "--numa-bind": runtime["numa_bind"], "--language-model-only": runtime["language_model_only"],
                "--enable-auto-tool-choice": True}
    for flag,value in expected.items():
        actual = flags.get(flag, False)
        if str(actual) != str(value):
            raise ValueError(f"Local model service {flag}={actual!r} does not match manifest value {value!r}")
    gpu_ids = [int(value) for value in (devices or "").split(",") if value]
    if gpu_ids != runtime["gpu_ids"] or len(gpu_ids) != len(set(gpu_ids)) or any(value < 0 for value in gpu_ids):
        raise ValueError(f"Local model service GPUs {gpu_ids!r} do not match manifest {runtime['gpu_ids']!r}")
    generation = json.loads(flags.get("--override-generation-config", "{}"))
    template = json.loads(flags.get("--default-chat-template-kwargs", "{}"))
    if generation != runtime["generation"] or template != {"enable_thinking": target["enable_thinking"]}:
        raise ValueError("Local model generation/thinking defaults differ from the manifest")
    return {"checked_launch_flags": expected, "gpu_ids": gpu_ids,
            "generation_defaults": generation, "chat_template_defaults": template}


class ContainerRuntime:
    def __init__(self, node: Path, python: Path):
        self.node, self.python = node.resolve(), python.resolve()
        self.python_root = self.python.parent.parent
        self.site_packages = ROOT / ".venv/lib/python3.12/site-packages"
        self.source_snapshot: Path | None = None
        self.mount_sources = [ROOT / "src", ROOT / "agents/pi", ROOT / "agents/deepseek-harness",
                              ROOT / "agents/mini-swe-agent/src",
                              ROOT / "agents/deepagents/libs/deepagents", self.site_packages,
                              self.node.parent.parent, self.python_root]
        for path in self.mount_sources:
            if not path.exists():
                raise FileNotFoundError(path)

    def mounts(self, agent_output: Path) -> list[dict]:
        return [{"type": "bind", "source": str(self.source_snapshot if path == ROOT / "src" and self.source_snapshot else path), "target": str(path), "read_only": True}
                for path in self.mount_sources] + [{"type": "bind", "source": str(agent_output.resolve()),
                            "target": "/hrp-output", "read_only": False}]

    def docker_mount_args(self, agent_output: Path) -> list[str]:
        args = []
        for mount in self.mounts(agent_output):
            args += ["--mount", f"type=bind,src={mount['source']},dst={mount['target']}" +
                     (",readonly" if mount["read_only"] else "")]
        return args

    async def launch(self, container: str, config: dict, output: Path, *, on_run_start=None) -> dict:
        async with asyncio.timeout(config.get("setup_timeout_sec", Trial._AGENT_SETUP_TIMEOUT_SEC)):
            config = {**config, "output_dir": "/hrp-output", "node_executable": str(self.node),
                      "harness_root": str(ROOT / "agents" / {"pi": "pi", "deepseek": "deepseek-harness",
                                                              "deepagents": "deepagents", "mini-swe-agent": "mini-swe-agent"}[config["harness"]])}
            write_json(output / "run.json", config)
            await command("docker", "exec", "-u", "root", container, "mkdir", "-p", "/hrp-home")
            user_args = ["-u", str(config["user"])] if config.get("user") is not None else []
            uid = (await command("docker", "exec", *user_args, container, "id", "-u")).strip()
            await command("docker", "exec", "-u", "root", container, "chown", uid, "/hrp-home")
            if int(uid) not in (0, os.getuid()):
                await command("docker", "exec", "-u", "root", container, "chown", f"{uid}:{os.getgid()}", "/hrp-output")
                await command("docker", "exec", "-u", "root", container, "chmod", "2770", "/hrp-output")
            detail = json.loads(await command("docker", "inspect", container))[0]
            task_env = dict(item.split("=", 1) for item in detail["Config"].get("Env", []) if "=" in item)
            task_path = task_env.get("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")
            write_json(output / "container_runtime.json", {"image_id": detail["Image"],
                "configured_image": detail["Config"].get("Image"), "agent_uid": int(uid),
                "cwd": config["cwd"], "home": "/hrp-home", "task_path": task_path,
                "networks": list(detail["NetworkSettings"]["Networks"]), "mounts": detail.get("Mounts", [])})
            args = ["docker", "exec", *user_args, "-w", config["cwd"], "-e", "HOME=/hrp-home",
                    "-e", "MSWEA_GLOBAL_CONFIG_DIR=/hrp-home/mini-swe-agent", "-e", "MSWEA_SILENT_STARTUP=1",
                    "-e", "PYTHONUNBUFFERED=1", "-e", f"PATH={self.node.parent}:{task_path}",
                    container]
            module = None
            if config["harness"] == "deepagents":
                module = "src.adapter.deepagents_native"
            elif config["harness"] == "mini-swe-agent":
                module = "src.adapter.mini_swe_native"
            else:
                args += [str(self.node), str(ROOT / f"src/adapter/{config['harness']}_native.mjs")]
            if module:
                # Agent Python dependencies must not contaminate the task's Python
                # environment, which may have a different Python version.
                paths = list(map(str, (ROOT, self.site_packages, ROOT / "agents/deepagents/libs/deepagents",
                                       ROOT / "agents/mini-swe-agent/src")))
                bootstrap = f"import sys,runpy;sys.path[:0]={paths!r};runpy.run_module({module!r},run_name='__main__')"
                args += [str(self.python), "-c", bootstrap]
            args += ["/hrp-output/run.json"]
        timed_out = False
        with (output / "agent.log").open("w") as log:
            if on_run_start:
                on_run_start()
            process = await asyncio.create_subprocess_exec(*args, stdout=log, stderr=log)
            try:
                code = await asyncio.wait_for(process.wait(), config["budget"]["wall_time_sec"])
            except TimeoutError:
                process.kill()
                await process.wait()
                # docker exec client termination does not reliably kill its child.
                # Stop the owned task before extracting/grading its final files.
                await command("docker", "stop", "--time", "2", container)
                await command("docker", "start", container)
                timed_out = True
                code = None
        await command("docker", "exec", "-u", "root", container, "chown", "-R", f"{os.getuid()}:{os.getgid()}", "/hrp-output")
        path = output / "result.json"
        result = ({"status": "generation_timeout"} if timed_out else json.loads(path.read_text()) if path.exists()
                  else {"status": "agent_error", "error": "runner did not produce result.json"})
        result["exit_code"] = code
        if not timed_out and (result.get("budget_reason") == "wall_time_sec"
                              or result.get("stop_reason") == "wall_time_sec"
                              or result.get("stopped_by") == "wall_time_sec"
                              or result.get("status") == "TimeExceeded"):
            # An adapter deadline may return before our supervisor fires.
            # Terminate its surviving tools before the verifier sees the files.
            await command("docker", "stop", "--time", "2", container)
            await command("docker", "start", container)
            result["native_status"] = result["status"]
            result["status"] = "generation_timeout"
        write_json(output / "process_result.json", result)
        return result


async def start_tb2(task_path: Path, output: Path, runtime: ContainerRuntime,
                    index: int, network_cidr: str = "10.242.0.0/16",
                    prepared_image: dict | None = None,
                    persistent_env: dict | None = None) -> tuple[Task, DockerEnvironment, str, str]:
    task = Task(task_path)
    if prepared_image:
        actual = (await command("docker", "image", "inspect", prepared_image["local_ref"], "--format", "{{.Id}}")).strip()
        if actual != prepared_image["image_id"]:
            raise ValueError("Task image no longer matches prepared inventory")
        task.config.environment.docker_image = prepared_image["local_ref"]
    paths = TrialPaths(output / "harbor")
    paths.mkdir()
    agent_dir = output / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    overlay = output / "network.json"
    subnet = ipaddress.ip_network(network_cidr)[index*16]
    compose = {"networks": {"default": {"ipam": {"config": [{"subnet": f"{subnet}/28"}]}}}}
    if prepared_image:
        compose["services"] = {"main": {"pull_policy": "never"}}
    write_json(overlay, compose)
    mounts = runtime.mounts(agent_dir)
    # Harbor 0.22 expects the caller (normally Trial) to add log mounts.
    for source, target in ((paths.agent_dir, "/logs/agent"), (paths.verifier_dir, "/logs/verifier"),
                           (paths.artifacts_dir, "/logs/artifacts")):
        mounts.append({"type": "bind", "source": str(source.resolve()), "target": target, "read_only": False})
    # Preserve the historical path only. New pinned images keep their own
    # distribution sources; Debian bookworm sources must not be added to Ubuntu.
    apt = ROOT / "src/benchmark/config/debian.sources"
    if apt.exists() and prepared_image is None:
        mounts.append({"type": "bind", "source": str(apt), "target": "/etc/apt/sources.list.d/debian.sources", "read_only": True})
    environment = DockerEnvironment(environment_dir=task.paths.environment_dir,
        environment_name=task.name, session_id="hrp-"+hashlib.sha256(str(output.resolve()).encode()).hexdigest()[:16], trial_paths=paths,
        task_env_config=task.config.environment, mounts=mounts,
        persistent_env=persistent_env if persistent_env is not None else {"UV_INSTALLER_GITHUB_BASE_URL": "https://ghfast.top/https://github.com",
                        "UV_PYTHON_INSTALL_MIRROR": "https://ghfast.top/https://github.com/astral-sh/python-build-standalone/releases/download"},
        extra_docker_compose=[overlay])
    try:
        await environment.start(force_build=False)
        environment.default_user = task.config.agent.user
        result = await environment._run_docker_compose_command(["ps", "-q", "main"])
        container = result.stdout.strip()
        if not container:
            raise RuntimeError("Harbor did not expose the main task container")
        cwd = (await environment.exec("pwd")).stdout.strip()
    except BaseException:
        try:
            await environment.stop(delete=True)
        except Exception as cleanup_error:
            write_json(output / "cleanup_error.json", {"error": repr(cleanup_error)})
        raise
    write_json(output / "task.json", {"task_path": str(task_path), "instruction": task.instruction,
                "task_config": task.config.model_dump(mode="json"), "cwd": cwd,
                "container_id": container, "harbor_version": "0.22.0"})
    return task, environment, container, cwd


async def grade_tb2(task: Task, environment: DockerEnvironment, output: Path) -> dict:
    try:
        verifier = Verifier(task, TrialPaths(output / "harbor"), environment)
        with environment.with_default_user(task.config.verifier.user):
            result = await asyncio.wait_for(verifier.verify(), task.config.verifier.timeout_sec)
        rewards = result.rewards or {}
        reward = float(rewards.get("reward", next(iter(rewards.values()), 0)))
        return {"resolved": reward > 0, "reward": reward, "status": "resolved" if reward > 0 else "failed",
                "verifier": result.model_dump(mode="json")}
    except TimeoutError:
        return {"resolved": False, "reward": 0., "status": "evaluation_timeout"}
    except Exception as error:
        return {"resolved": False, "reward": 0., "status": "infrastructure_error", "error": repr(error)}


async def container_gateway(container: str) -> str:
    detail = json.loads(await command("docker", "inspect", container))[0]
    for network in detail["NetworkSettings"]["Networks"].values():
        if network.get("Gateway"):
            return network["Gateway"]
        # Internal Docker networks omit a container default route, but their
        # bridge host remains reachable at the IPAM gateway address.
        definition = json.loads(await command("docker", "network", "inspect", network["NetworkID"]))[0]
        for block in definition.get("IPAM", {}).get("Config", []):
            if block.get("Gateway"):
                return block["Gateway"]
    raise RuntimeError("Task network has no reachable bridge host address for the model gateway")
