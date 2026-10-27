"""Frozen next-decision sampling through each framework's original decoder."""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
import harbor

from src.benchmark.native_gateway import write_json
from src.hrp.native_measurement import fingerprint
from src.adapter.terminal_actions import analyze_bash_command

ROOT = Path(__file__).resolve().parents[2]


def native_boundary(record: dict, state: dict) -> dict:
    """Read native lineage; unknown historical parent decisions stay unknown."""
    harness = record.get("identity", {}).get("harness", state.get("harness"))
    controller = state.get("controller") or {}
    kind = record.get("call_kind", state.get("call_kind", "unclassified"))
    parent = None
    if harness == "deepagents":
        namespace = state.get("metadata", {}).get("langgraph_checkpoint_ns", "")
        parts = namespace.split("|")
        agent = "|".join(parts[:-1]) or "root"
        parent = ("|".join(parts[:-2]) or "root") if len(parts) > 1 else None
    elif harness == "deepseek":
        header = (state.get("session") or {}).get("header", {})
        agent, parent = header.get("id"), header.get("parentSession")
    elif harness == "pi":
        agent = controller.get("session_id", "root")
    elif harness == "terminus-2" and any(k in kind for k in ("summary", "summar")):
        agent, parent = kind+":"+str(controller.get("summarization_count", 0)), "root"
    else:
        agent = "root"
    trace = {"agent_id": agent, "parent_agent_id": parent,
             "agent_role": "subagent" if parent is not None else "root",
             "parent_state_id": None, "parent_tool_run_id": None,
             "previous_state_id": None, **state.get("trace", {})}
    return {**trace, "call_id": record.get("attempt"), "logical_call_id": record.get("state_id"),
            "call_kind": kind, "request_fingerprint": fingerprint(record.get("request", {})),
            "native_state_fingerprint": fingerprint(state), "cwd": state.get("cwd"),
            "pending_completion": controller.get("pending_completion", False),
            "started": record.get("started"), "elapsed_sec": record.get("elapsed_sec"),
            "overlapping_attempts_at_start": record.get("overlapping_attempts_at_start"),
            "lineage_source": "native_trace" if state.get("trace") else "historical_native_metadata",
            "full_controller_restored": False}


def annotate_shell_decisions(response: dict) -> dict:
    """Attach ordered AST projections without replacing the complete decision."""
    for choice in response.get("choices", []):
        for call in (choice.get("message") or {}).get("tool_calls") or []:
            function = call.get("function", {})
            if function.get("name") not in ("bash", "execute"):
                continue
            args = function.get("arguments", {})
            try:
                args = json.loads(args) if isinstance(args, str) else args
            except (ValueError, TypeError):
                continue
            if isinstance(args, dict) and isinstance(args.get("command"), str):
                call["bash_projection"] = analyze_bash_command(args["command"])
    return response


def native_response(harness: str, result: dict) -> dict:
    """Lossless native result is retained alongside this diagnostic projection."""
    if harness in ("terminus-2", "mini-swe-agent"):
        response = (result.get("response", {}) if harness == "terminus-2" else
                    result.get("message", {}).get("extra", {}).get("response", {}))
        response = {**response, "native_decoder": harness, "native_result": result}
        if result.get("error"):
            response["error"] = result["error"]
        if result.get("parser_error") or response.get("native_parse", {}).get("error"):
            response["native_parser_error"] = result.get("parser_error") or response["native_parse"]["error"]
        return annotate_shell_decisions(response)
    if harness == "deepagents":
        message = result.get("message", {})
        calls = [{"id": call.get("id"), "type": "function", "function": {
            "name": call.get("name"), "arguments": call.get("args")}}
            for call in message.get("tool_calls", []) + message.get("invalid_tool_calls", [])]
        finish = message.get("response_metadata", {}).get("finish_reason")
        projected = {"content": message.get("content"), "tool_calls": calls}
        usage = message.get("usage_metadata", {})
    else:
        message = result.get("message", {}) if harness == "pi" else result.get("decision", {})
        blocks = message.get("content", [])
        calls = [{"id": block.get("id"), "type": "function", "function": {
            "name": block.get("name"), "arguments": block.get("arguments")}}
            for block in blocks if block.get("type") in ("toolCall", "tool-call")]
        projected = {"content": "\n".join(block.get("text", "") for block in blocks if block.get("type") == "text"),
                     "tool_calls": calls}
        if harness == "pi":
            finish = {"length": "length", "toolUse": "tool_calls", "stop": "stop"}.get(message.get("stopReason"), message.get("stopReason"))
        else:
            finish = {"max-tokens": "length", "tool-calls": "tool_calls"}.get(
                message.get("finish_reason", {}).get("kind"), message.get("finish_reason", {}).get("kind"))
        usage = message.get("usage", {})
    response = {"choices": [{"message": projected, "finish_reason": finish}], "usage": usage,
                "native_decoder": harness, "native_result": result}
    if result.get("error") or result.get("status") in ("error", "agent_error", "infrastructure_error") or message.get("classification") == "provider_error":
        response["error"] = result.get("error") or {"status": result.get("status"), "native_decision": message}
    return annotate_shell_decisions(response)


async def sample_native(*, gateway, record: dict, state_path: Path, output: Path,
                        model_role: str, seed: int, node: Path, sampling_budget: dict | None = None) -> dict:
    harness = record["identity"]["harness"]
    source_bytes = state_path.read_bytes()
    source_state = json.loads(source_bytes)
    model = gateway.models[model_role]
    output.mkdir(parents=True, exist_ok=False)
    native_output = output / "native"
    native_output.mkdir()
    budget = {"max_steps": 1, **(sampling_budget or {"max_model_calls": 8,
              "max_output_tokens": 8*32768, "wall_time_sec": 1200})}
    token, session = gateway.register(output / "transport", identity={**record["identity"],
        "operation": "sample", "source_state_id": record["state_id"], "role": model_role},
        budget=budget, snapshots=False, model_role=model_role)
    session.update(frozen_request=record["request"], sample_seed=seed)
    selected = {**model, "api_base": f"http://127.0.0.1:{gateway.port}/s/{token}/v1", "api_key": "EMPTY"}
    selected["max_tokens"] = record["request"].get("max_tokens", record["request"].get("max_completion_tokens", model["max_tokens"]))
    config = {"operation": "sample", "state_path": str(state_path.resolve()),
        "cwd": str(output.resolve()), "output_dir": str(native_output.resolve()),
        "harness_root": str(ROOT / "agents" / {"pi": "pi", "deepseek": "deepseek-harness", "deepagents": "deepagents",
                                              "mini-swe-agent": "mini-swe-agent", "terminus-2": "terminus-2"}[harness]),
        "model": selected, "budget": budget, "seed": seed}
    config_path = output / "config.json"
    if harness == "terminus-2":
        config["harness_root"] = str(Path(harbor.__file__).parent / "agents/terminus_2")
    write_json(config_path, config)
    implementation = ROOT
    for ancestor in state_path.parents:
        if (ancestor / "implementation/src").is_dir():
            implementation = ancestor / "implementation"
            break
    modules = {"deepagents": "deepagents_native", "mini-swe-agent": "mini_swe_native", "terminus-2": "terminus_native"}
    module = "src.adapter."+modules[harness] if harness in modules else None
    command = ([sys.executable, "-c", f"import sys,runpy;sys.path.insert(0,{str(implementation)!r});runpy.run_module({module!r},run_name='__main__')"] if module else
               [str(node), str(implementation / f"src/adapter/{harness}_native.mjs")]) + [str(config_path.resolve())]
    try:
        async with gateway.native_workers:
            queue_wait = time.time()-session["started"]
            # Waiting for an available sampler process is scheduling cost, not
            # time consumed by this model decision. Start its cap on admission.
            session["started"] = time.time()
            with (output / "runner.log").open("w") as log:
                env = {**os.environ, "MSWEA_GLOBAL_CONFIG_DIR": str(native_output / "global-config"), "MSWEA_SILENT_STARTUP": "1"}
                process = await asyncio.create_subprocess_exec(*command, cwd=ROOT, stdout=log, stderr=log, env=env)
                try:
                    code = await asyncio.wait_for(process.wait(), budget["wall_time_sec"]+10)
                except TimeoutError:
                    process.kill()
                    await process.wait()
                    raise
        result_path = native_output / "result.json"
        if not result_path.exists():
            raise RuntimeError(f"Native sampler exited {code}; see {output / 'runner.log'}")
        native = json.loads(result_path.read_text())
        attempts = [json.loads(path.read_text()) for path in sorted((output / "transport/wire").glob("*/record.json"))]
        unchanged = state_path.read_bytes() == source_bytes
        if not unchanged:
            raise ValueError("Native sampling modified its source state")
        comparable = bool(attempts) and all(all(attempt["request"].get(key) == record["request"].get(key)
                            for key in ("messages", "tools", "tool_choice")) for attempt in attempts)
        if not comparable:
            raise ValueError("Native sampling failed frozen-context/tool equivalence")
        private_replay = any(message.get("reasoning_content") or message.get("reasoning_details")
                             for message in record["request"].get("messages", []))
        response = native_response(harness, native)
        response["native_call_kind"] = record.get("call_kind", source_state.get("call_kind", "decision"))
        wire_response = attempts[-1].get("response", {})
        response["transport_response"] = wire_response
        wire_finish = (wire_response.get("choices") or [{}])[0].get("finish_reason")
        if harness in ("terminus-2", "mini-swe-agent"):
            if response.get("error") and attempts[-1].get("http_status") == 200:
                # A native decoder exception (including output-limit errors)
                # is not a provider HTTP failure. Keep both records and flags.
                response["native_decoder_error"] = response.pop("error")
                response["native_decoder_failed"] = True
                response["native_parser_error"] = str(response["native_decoder_error"])
            if not response.get("choices"):
                response["choices"] = json.loads(json.dumps(wire_response.get("choices", [])))
            if wire_finish is not None:
                for choice in response.get("choices", []):
                    choice["finish_reason"] = wire_finish
        return {"http_status": attempts[-1].get("http_status"), "response": response,
                "native_result": native, "request": attempts[-1]["request"],
                "request_fingerprint": fingerprint(attempts[-1]["request"]),
                "native_decoding": True, "http_attempts": len(attempts),
                "cost": json.loads((output / "transport/cost.json").read_text()),
                "source_unchanged": unchanged, "same_visible_context_and_tools": comparable,
                "worker_queue_wait_sec": queue_wait,
                "cross_model_private_replay_unverified": bool(private_replay or native.get("replay_degradations")),
                "sampling_scope": "native provider decoder; no controller prefix/tool execution",
                "elapsed_sec": sum(attempt.get("elapsed_sec", 0) for attempt in attempts)}
    finally:
        gateway.sessions.pop(token, None)
