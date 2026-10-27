"""One recording gateway and resource ledger shared by both benchmarks.

The native clients still own parsing, retries and termination. Every HTTP
attempt, including failed attempts, is separately recorded and charged.
"""
from __future__ import annotations

import asyncio
import copy
import json
import secrets
import time
from pathlib import Path

import aiohttp
from aiohttp import web

from src.hrp.native_measurement import decode_response, fingerprint


async def command(*args: str, timeout: float = 1800) -> str:
    process = await asyncio.create_subprocess_exec(*map(str, args),
              stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
    except TimeoutError:
        process.kill()
        await process.communicate()
        raise
    if process.returncode:
        raise RuntimeError(f"{args[0]} exited {process.returncode}: {stderr.decode()[-4000:]}")
    return stdout.decode()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n")
    temporary.replace(path)


class NativeGateway:
    def __init__(self, target: dict, teacher: dict | None, *, local_concurrency: int,
                 teacher_concurrency: int, port: int = 0, native_worker_concurrency: int = 40,
                 openrouter_max_retries: int = 10, openrouter_retry_initial_sec: float = 1.,
                 openrouter_retry_max_sec: float = 30.):
        self.models = {"target": target, "teacher": teacher}
        self.limits = {"target": asyncio.Semaphore(local_concurrency),
                       "teacher": asyncio.Semaphore(teacher_concurrency)}
        self.port = port
        self.sessions: dict[str, dict] = {}
        self.active = {"target": 0, "teacher": 0}
        self.peak = {"target": 0, "teacher": 0}
        self.native_workers = asyncio.Semaphore(min(native_worker_concurrency, local_concurrency + teacher_concurrency))
        self.openrouter_max_retries = openrouter_max_retries
        self.openrouter_retry_initial_sec = openrouter_retry_initial_sec
        self.openrouter_retry_max_sec = openrouter_retry_max_sec

    @staticmethod
    def _is_openrouter(model: dict) -> bool:
        return model.get("backend") == "openrouter" or "openrouter.ai" in model.get("api_base", "")

    @staticmethod
    def _retryable_status(status: int) -> bool:
        return status in {408, 409, 425, 429} or status >= 500

    @staticmethod
    def _retryable_transport_error(error: Exception) -> bool:
        return isinstance(error, (aiohttp.ClientConnectionError, aiohttp.ClientPayloadError,
                                  asyncio.TimeoutError)) and not isinstance(
                                      error, aiohttp.ClientConnectorCertificateError)

    def _retry_delay(self, retry_number: int, retry_after: str | None = None) -> float:
        try:
            delay = float(retry_after) if retry_after is not None else (
                self.openrouter_retry_initial_sec * 2 ** (retry_number - 1))
        except ValueError:
            delay = self.openrouter_retry_initial_sec * 2 ** (retry_number - 1)
        return max(0., min(delay, self.openrouter_retry_max_sec))

    def _retry_allowed(self, session: dict, retry_number: int) -> bool:
        if retry_number > self.openrouter_max_retries:
            return False
        budget = session["budget"]
        return not any(budget.get(key) is not None and used >= budget[key]
                       for key, used in (("max_model_calls", session["calls"]),
                           ("wall_time_sec", time.time() - session["started"])))

    async def start(self) -> None:
        self.client = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1200), trust_env=True)
        app = web.Application(client_max_size=64*1024*1024)
        app.router.add_post("/s/{token}/tokenize", self.tokenize)
        app.router.add_route("*", "/s/{token}/v1/{tail:.*}", self.handle)
        self.runner = web.AppRunner(app, access_log=None)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, "0.0.0.0", self.port)
        await self.site.start()
        self.port = self.site._server.sockets[0].getsockname()[1]

    async def close(self) -> None:
        await self.runner.cleanup()
        await self.client.close()

    async def tokenize(self, request: web.Request) -> web.Response:
        if request.match_info["token"] not in self.sessions:
            raise web.HTTPNotFound()
        target = self.models["target"]
        url = target["api_base"].removesuffix("/v1").rstrip("/") + "/tokenize"
        async with self.client.post(url, json=await request.json()) as response:
            return web.Response(body=await response.read(), status=response.status, content_type="application/json")

    def register(self, output: Path, *, identity: dict, budget: dict,
                 snapshots: bool, container_id: str | None = None,
                 model_role: str = "target") -> tuple[str, dict]:
        token = secrets.token_urlsafe(24)
        session = {"output": output, "identity": identity, "budget": budget,
                   "model_role": model_role,
                   "snapshots": snapshots, "container_id": container_id,
                   "started": time.time(), "calls": 0, "requests_seen": 0, "output_tokens": 0,
                   "input_tokens": 0, "usage_missing_calls": 0,
                   "provider_cost_usd_reported": 0., "provider_cost_missing_calls": 0,
                   "active_calls": 0, "snapshot_lock": asyncio.Lock()}
        self.sessions[token] = session
        return token, session

    async def snapshot(self, session: dict, attempt: str) -> dict:
        if not session["snapshots"] or not session.get("container_id"):
            return {"status": "not_captured", "full_process_restore": False}
        async with session["snapshot_lock"]:
            try:
                container = session["container_id"]
                detail = json.loads(await command("docker", "inspect", container))[0]
                tag = "hrp-state:" + fingerprint({"container": container, "attempt": attempt})[:24]
                image = (await command("docker", "commit", container, tag)).strip().splitlines()[-1]
                mounts = [{k: m.get(k) for k in ("Type", "Destination", "RW", "Source")}
                          for m in detail.get("Mounts", [])]
                mutable_mounts = [m for m in mounts if m["RW"] and not m["Destination"].startswith(("/logs", "/hrp-output"))]
                return {"status": "filesystem_snapshot", "image": image, "tag": tag,
                        "container_id": container, "mounts": mounts,
                        "task_filesystem_restorable": not mutable_mounts,
                        "excluded_mutable_mounts": mutable_mounts,
                        "concurrent_model_calls": session["active_calls"],
                        "full_process_restore": False,
                        "limitations": ["Docker commit preserves filesystem, not running process memory or external network state.",
                                        "Parallel native tools/subagents may mutate state after this boundary."]}
            except Exception as error:
                return {"status": "unavailable", "error": str(error), "full_process_restore": False}

    def save_cost(self, session: dict) -> None:
        write_json(session["output"] / "cost.json", {
            "http_attempts": session["calls"], "input_tokens": session["input_tokens"],
            "native_gateway_requests": session["requests_seen"],
            "output_tokens": session["output_tokens"], "wall_time_sec": time.time()-session["started"],
            "cost_convention": "all wire attempts including summary, subagents and retries",
            "usage_missing_calls": session["usage_missing_calls"],
            "provider_cost_usd_reported": session["provider_cost_usd_reported"],
            "provider_cost_missing_calls": session["provider_cost_missing_calls"],
            "token_budget_enforcement": "at call boundaries; in-flight calls can overshoot; missing usage is not reported as measured zero"})

    async def handle(self, request: web.Request) -> web.StreamResponse:
        session = self.sessions.get(request.match_info["token"])
        if session is None:
            raise web.HTTPNotFound()
        role = session["model_role"]
        target = self.models[role]
        tail = request.match_info["tail"]
        if tail == "tokenize":
            return await self.tokenize(request)
        if request.method == "GET" and tail == "models":
            return web.json_response({"object": "list", "data": [{"id": target["id"], "object": "model"}]})
        if request.method != "POST" or tail not in ("chat/completions", "responses"):
            raise web.HTTPNotFound()
        payload = await request.json()
        client_payload = None
        if "frozen_request" in session:
            client_payload = payload
            payload = copy.deepcopy(session["frozen_request"])
            payload["model"] = target["id"]
            payload["seed"] = session["sample_seed"]
            # Preserve the native consumer's transport mode; context/tools and
            # native per-call output limits come from the frozen actual request.
            payload["stream"] = client_payload.get("stream", False)
            if not payload["stream"]:
                payload.pop("stream_options", None)
            if role == "teacher":
                payload.pop("chat_template_kwargs", None)
                if target.get("temperature") is None:
                    payload.pop("temperature", None)
                else:
                    payload["temperature"] = target["temperature"]
                payload["reasoning"] = {"effort": target.get("reasoning_effort", "medium")}
        elif role == "teacher":
            # Independent teacher trajectories use the same remote model policy
            # as probes, including summary/subagent calls made by native loops.
            payload["model"] = target["id"]
            payload.pop("chat_template_kwargs", None)
            payload.pop("reasoning_effort", None)
            if target.get("temperature") is None:
                payload.pop("temperature", None)
            else:
                payload["temperature"] = target["temperature"]
            payload["reasoning"] = {**(payload.get("reasoning") or {}),
                                    "effort": target.get("reasoning_effort", "medium")}
        session["requests_seen"] += 1
        attempt = f"call-{session['requests_seen']:06d}"
        state_id = request.headers.get("X-HRP-State-ID") or fingerprint(payload)
        kind = request.headers.get("X-HRP-Call-Kind", "unclassified")
        directory = session["output"] / "wire" / attempt
        directory.mkdir(parents=True, exist_ok=False)
        over_budget = any(session["budget"].get(key) is not None and used >= session["budget"][key]
                          for key, used in (("max_model_calls", session["calls"]),
                              ("max_output_tokens", session["output_tokens"]),
                              ("wall_time_sec", time.time() - session["started"])))
        # This is an explicit common experimental resource cap, not a native stop.
        if over_budget:
            write_json(directory / "record.json", {"state_id": state_id, "status": "budget_exhausted",
                       "request": payload, "call_kind": kind, "charged_http_attempt": False})
            self.save_cost(session)
            return web.json_response({"error": {"message": "HRP experiment resource budget exhausted",
                                     "type": "hrp_budget_exhausted"}}, status=400)
        snapshot = await self.snapshot(session, attempt)
        record = {"state_id": state_id, "attempt": attempt, "call_kind": kind,
                  "request": payload, "request_fingerprint": fingerprint(payload),
                  "identity": session["identity"], "environment": snapshot,
                  "started": time.time(), "overlapping_attempts_at_start": sorted(session.get("active_attempts", set())),
                  "budget_before": {
                      "model_calls": session["calls"], "output_tokens": session["output_tokens"],
                      "elapsed_sec": time.time()-session["started"]},
                  "replay_scope": "frozen provider request; not an in-flight controller restore"}
        if client_payload is not None:
            record["native_client_request_before_frozen_replay"] = client_payload
        write_json(directory / "record.json", record)
        session["active_calls"] += 1
        session.setdefault("active_attempts", set()).add(attempt)
        usage_recorded = cost_recorded = False
        try:
            async with self.limits[role]:
                self.active[role] += 1
                self.peak[role] = max(self.peak[role], self.active[role])
                try:
                    url = target["api_base"].rstrip("/") + "/" + tail
                    headers = {"Authorization": "Bearer " + target.get("api_key", "EMPTY")}
                    transport_attempts = record.setdefault("transport_attempts", [])
                    retry_number = 0
                    while True:
                        wire_started = time.time()
                        downstream_prepared = False
                        session["calls"] += 1
                        try:
                            async with self.client.post(url, json=payload, headers=headers) as upstream:
                                should_retry = (self._is_openrouter(target) and
                                    self._retryable_status(upstream.status) and
                                    self._retry_allowed(session, retry_number + 1))
                                wire = {"index": len(transport_attempts) + 1,
                                        "http_status": upstream.status,
                                        "elapsed_sec": time.time() - wire_started,
                                        "retry": should_retry}
                                if should_retry:
                                    await upstream.read()
                                    retry_number += 1
                                    delay = self._retry_delay(retry_number, upstream.headers.get("Retry-After"))
                                    wire["retry_delay_sec"] = delay
                                    transport_attempts.append(wire)
                                    session["usage_missing_calls"] += 1
                                    session["provider_cost_missing_calls"] += int(role == "teacher")
                                    write_json(directory / "record.json", record)
                                    self.save_cost(session)
                                    await asyncio.sleep(delay)
                                    continue
                                transport_attempts.append(wire)
                                response = web.StreamResponse(status=upstream.status,
                                    headers={"Content-Type": upstream.headers.get("Content-Type", "application/json")})
                                await response.prepare(request)
                                downstream_prepared = True
                                chunks = []
                                with (directory / "response.raw").open("wb") as stream:
                                    async for chunk in upstream.content.iter_any():
                                        chunks.append(chunk)
                                        stream.write(chunk)
                                        await response.write(chunk)
                                raw = b"".join(chunks).decode(errors="replace")
                                record["http_status"] = upstream.status
                                record["status"] = "completed" if upstream.status == 200 else "provider_error"
                                try:
                                    decoded = decode_response(raw, streaming="text/event-stream" in upstream.headers.get("Content-Type", ""))
                                    record["response"] = decoded
                                    usage = decoded.get("usage", {})
                                    usage_recorded = bool(usage)
                                    if usage.get("cost") is not None:
                                        session["provider_cost_usd_reported"] += float(usage["cost"])
                                        cost_recorded = True
                                    session["output_tokens"] += usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
                                    session["input_tokens"] += usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
                                except (ValueError, TypeError) as error:
                                    record["parse_error"] = str(error)
                                await response.write_eof()
                                return response
                        except Exception as error:
                            should_retry = (not downstream_prepared and self._is_openrouter(target) and
                                self._retryable_transport_error(error) and
                                self._retry_allowed(session, retry_number + 1))
                            wire = {"index": len(transport_attempts) + 1,
                                    "status": "transport_error", "error": repr(error),
                                    "elapsed_sec": time.time() - wire_started,
                                    "retry": should_retry}
                            transport_attempts.append(wire)
                            if not should_retry:
                                raise
                            retry_number += 1
                            delay = self._retry_delay(retry_number)
                            wire["retry_delay_sec"] = delay
                            session["usage_missing_calls"] += 1
                            session["provider_cost_missing_calls"] += int(role == "teacher")
                            write_json(directory / "record.json", record)
                            self.save_cost(session)
                            await asyncio.sleep(delay)
                finally:
                    self.active[role] -= 1
        except Exception as error:
            record.update(status="transport_error", error=str(error))
            raise
        finally:
            session["active_calls"] -= 1
            session["active_attempts"].discard(attempt)
            session["usage_missing_calls"] += int(not usage_recorded)
            session["provider_cost_missing_calls"] += int(role == "teacher" and not cost_recorded)
            record["elapsed_sec"] = time.time() - record["started"]
            write_json(directory / "record.json", record)
            self.save_cost(session)

    async def sample(self, record: dict, role: str, *, seed: int) -> dict:
        """Copy an already rendered native request; never invoke agent/controller."""
        model = self.models[role]
        if model is None:
            raise ValueError(f"No {role} model configured")
        payload = copy.deepcopy(record["request"])
        payload["model"] = model["id"]
        payload["stream"] = False
        payload.pop("stream_options", None)
        payload["seed"] = seed
        if role == "teacher":
            if model.get("temperature") is None:
                payload.pop("temperature", None)
            else:
                payload["temperature"] = model["temperature"]
            # Provider-specific generation options are sampling controls; context
            # and tool definitions are untouched. No reasoning-token budget.
            payload.pop("chat_template_kwargs", None)
            payload["reasoning"] = {"effort": model.get("reasoning_effort", "medium")}
        request_hash = fingerprint(payload)
        started = time.time()
        async with self.limits[role]:
            self.active[role] += 1
            self.peak[role] = max(self.peak[role], self.active[role])
            try:
                attempts = []
                retry_number = 0
                while True:
                    wire_started = time.time()
                    try:
                        async with self.client.post(model["api_base"].rstrip("/") + "/chat/completions",
                                   json=payload, headers={"Authorization": "Bearer " + model.get("api_key", "EMPTY")}) as response:
                            raw = await response.text()
                            should_retry = (self._is_openrouter(model) and
                                self._retryable_status(response.status) and
                                retry_number < self.openrouter_max_retries)
                            wire = {"index": len(attempts) + 1, "http_status": response.status,
                                    "elapsed_sec": time.time() - wire_started, "retry": should_retry}
                            attempts.append(wire)
                            if should_retry:
                                retry_number += 1
                                delay = self._retry_delay(retry_number, response.headers.get("Retry-After"))
                                wire["retry_delay_sec"] = delay
                                await asyncio.sleep(delay)
                                continue
                            try:
                                decoded = json.loads(raw)
                            except ValueError:
                                decoded = {"error": {"raw_response": raw}}
                            usage = decoded.get("usage") or {}
                            failed_attempts = sum(attempt.get("http_status") != 200 for attempt in attempts)
                            missing_success = int(not usage and response.status == 200)
                            cost = {"http_attempts": len(attempts),
                                    "input_tokens": usage.get("prompt_tokens", 0) or 0,
                                    "output_tokens": usage.get("completion_tokens", 0) or 0,
                                    "usage_missing_calls": failed_attempts + missing_success,
                                    "provider_cost_usd_reported": usage.get("cost") or 0.,
                                    "provider_cost_missing_calls": int(role == "teacher") * (
                                        failed_attempts + int(usage.get("cost") is None and response.status == 200))}
                            return {"http_status": response.status, "response": decoded,
                                    "cost": cost, "transport_attempts": attempts,
                                    "request": payload, "request_fingerprint": request_hash,
                                    "elapsed_sec": time.time()-started,
                                    "sampling_scope": "same native wire context/tools; no tools executed"}
                    except (aiohttp.ClientError, TimeoutError) as error:
                        should_retry = (self._is_openrouter(model) and
                            self._retryable_transport_error(error) and
                            retry_number < self.openrouter_max_retries)
                        wire = {"index": len(attempts) + 1, "status": "transport_error",
                                "error": repr(error), "elapsed_sec": time.time() - wire_started,
                                "retry": should_retry}
                        attempts.append(wire)
                        if should_retry:
                            retry_number += 1
                            delay = self._retry_delay(retry_number)
                            wire["retry_delay_sec"] = delay
                            await asyncio.sleep(delay)
                            continue
                        return {"http_status": None, "response": {"error": {"message": repr(error)}},
                                "request": payload, "request_fingerprint": request_hash,
                                "elapsed_sec": time.time()-started, "transport_attempts": attempts,
                                "cost": {"http_attempts": len(attempts), "input_tokens": 0, "output_tokens": 0,
                                         "usage_missing_calls": len(attempts), "provider_cost_usd_reported": 0.,
                                         "provider_cost_missing_calls": int(role == "teacher") * len(attempts)},
                                "sampling_scope": "shared-Bash raw response; failed transport retained"}
            finally:
                self.active[role] -= 1
