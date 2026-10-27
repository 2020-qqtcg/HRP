"""Observe Harbor's actual Terminus-2 loop without replacing its terminal loop."""
from __future__ import annotations

import asyncio
import dataclasses
import inspect
import json
import sys
import time
from pathlib import Path

from harbor.agents.terminus_2.terminus_2 import Terminus2
from harbor.agents.terminus_2.terminus_json_plain_parser import TerminusJSONPlainParser
from harbor.llms.lite_llm import LiteLLM
from harbor.models.agent.context import AgentContext
from harbor.trial.trial import Trial

from src.benchmark.native_gateway import write_json
from src.hrp.native_measurement import fingerprint


def llm_options(model: dict) -> dict:
    return {"model_name": "openai/"+model["id"].removeprefix("openai/"),
        "api_base": model["api_base"], "temperature": model.get("temperature"),
        "model_info": {"max_input_tokens": model["context_window"], "max_output_tokens": model["max_tokens"],
                       "input_cost_per_token": 0, "output_cost_per_token": 0}}


def project_response(content: str, *, call_kind: str = "main") -> dict:
    """Keep keystrokes, waits, ordering and completion separate from Bash tools."""
    if not call_kind.startswith("main"):
        return {"choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                "native_call_kind": call_kind}
    parsed = TerminusJSONPlainParser().parse_response(content)
    calls = [{"id": f"command-{i}", "type": "function", "function": {
        "name": "terminal_keystrokes", "arguments": dataclasses.asdict(command)}}
        for i, command in enumerate(parsed.commands)]
    # task_complete affects the controller even when commands are also present.
    if not parsed.error:
        calls.append({"id": "completion-flag", "type": "function", "function": {
            "name": "terminal_completion_flag", "arguments": {"task_complete": parsed.is_task_complete}}})
    response = {"choices": [{"message": {"role": "assistant", "content": content, "tool_calls": calls},
                              "finish_reason": "stop"}], "native_parse": dataclasses.asdict(parsed),
                "native_call_kind": call_kind,
                # Measurement schemas for the native JSON parser's output.
                # These are never sent to the model as extra tools.
                "decision_tool_definitions": [
                    {"type": "function", "function": {"name": "terminal_keystrokes", "parameters": {
                        "type": "object", "properties": {"keystrokes": {"type": "string"}, "duration": {"type": "number"}},
                        "required": ["keystrokes", "duration"], "additionalProperties": False}}},
                    {"type": "function", "function": {"name": "terminal_completion_flag", "parameters": {
                        "type": "object", "properties": {"task_complete": {"type": "boolean"}},
                        "required": ["task_complete"], "additionalProperties": False}}}]}
    if parsed.error:
        response["native_parser_error"] = parsed.error
    return response


async def run_terminus(environment, config: dict, output: Path, *, on_run_start=None) -> dict:
    write_json(output / "run.json", config)
    profile = config.get("profile", {}).get("options", {})
    agent = Terminus2(logs_dir=output, **llm_options(config["model"]),
        max_turns=config["budget"]["max_steps"], llm_kwargs={"api_key": "EMPTY"},
        llm_call_kwargs={"max_tokens": config["model"]["max_tokens"], "seed": config["seed"]}, **profile)
    original = agent._llm.call
    calls = 0
    previous_state = None
    previous_root_state = None
    attempts_by_episode = {}
    started = time.time()

    async def observed(prompt, message_history=None, **kwargs):
        nonlocal calls, previous_state, previous_root_state
        calls += 1
        frames = inspect.stack(context=0)
        subagent = next((frame.frame.f_locals.get("agent_name") for frame in frames if frame.function == "_run_subagent"), None)
        kind = str(subagent) if subagent else "main"
        # The fallback short summary calls _llm directly, outside _run_subagent.
        if not subagent and any(frame.function == "_query_llm" and
                prompt == frame.frame.f_locals.get("short_prompt") for frame in frames):
            kind = "short_summary"
        del frames
        if kind == "main":
            attempt = attempts_by_episode.get(agent._n_episodes, 0)
            attempts_by_episode[agent._n_episodes] = attempt + 1
            if attempt:
                kind = "main_retry"
        history = message_history or []
        serialized_history = [message.model_dump(mode="json") if hasattr(message, "model_dump") else message for message in history]
        state_id = fingerprint([calls, prompt, serialized_history])
        is_summary = not kind.startswith("main")
        trace = {"agent_id": kind+":"+str(agent._summarization_count) if is_summary else "root",
                 "parent_agent_id": "root" if is_summary else None,
                 "agent_role": "subagent" if is_summary else "root",
                 "parent_state_id": previous_root_state if is_summary else None,
                 "parent_relation": "context_maintenance" if is_summary else None,
                 "previous_state_id": previous_state}
        previous_state = state_id
        if not is_summary:
            previous_root_state = state_id
        session = agent._session
        write_json(output / "states" / f"{state_id}.json", {
            "state_id": state_id, "harness": "terminus-2", "call_kind": kind,
            "trace": trace,
            "prompt": prompt, "message_history": serialized_history,
            "call_kwargs": {k: str(v) if isinstance(v, Path) else v for k,v in kwargs.items()},
            "controller": {"episodes": agent._n_episodes, "pending_completion": agent._pending_completion,
                "summarization_count": agent._summarization_count, "elapsed_sec": time.time()-started,
                "previous_terminal_buffer": session._previous_buffer if session else None,
                "terminal_markers": session._markers if session else [],
                "pending_handoff_prompt": agent._pending_handoff_prompt},
            "cwd": config["cwd"], "budget": config["budget"],
            "restore_scope": "frozen next model request; tmux/process state is NOT restorable"})
        return await original(prompt, message_history=history, **{**kwargs, "extra_headers": {
            **kwargs.get("extra_headers", {}), "X-HRP-State-ID": state_id, "X-HRP-Call-Kind": kind}})

    agent._llm.call = observed
    context = AgentContext()
    phase = "setup"
    try:
        async with asyncio.timeout(config.get("setup_timeout_sec", Trial._AGENT_SETUP_TIMEOUT_SEC)):
            await agent.setup(environment)
        phase = "generation"
        started = time.time()
        if on_run_start:
            on_run_start()
        async with asyncio.timeout(config["budget"]["wall_time_sec"]):
            await agent.run(config["instruction"], environment, context)
        result = {"status": "native_loop_returned", "native_context": context.model_dump(mode="json")}
    except TimeoutError:
        result = {"status": f"{phase}_timeout", "native_context": context.model_dump(mode="json")}
    except Exception as error:
        result = {"status": "agent_error", "error": repr(error), "native_context": context.model_dump(mode="json")}
    result.update(top_level_steps=agent._n_episodes, observed_llm_boundaries=calls,
                  native_episode_guard=agent._max_episodes)
    write_json(output / "result.json", result)
    return result


async def sample(config: dict) -> dict:
    state = json.loads(Path(config["state_path"]).read_text())
    model = LiteLLM(**llm_options(config["model"]), api_key="EMPTY")
    kwargs = {k:v for k,v in state["call_kwargs"].items() if k not in ("logging_path", "extra_headers")}
    kwargs.update(max_tokens=config["model"]["max_tokens"], seed=config["seed"])
    response = await model.call(state["prompt"], message_history=state["message_history"], **kwargs)
    return {"status": "sampled", "response": project_response(response.content, call_kind=state["call_kind"]),
            "native_response": {"content": response.content, "reasoning_content": response.reasoning_content,
                                "usage": response.usage.model_dump(mode="json") if response.usage else None}}


def main():
    config = json.loads(Path(sys.argv[1]).read_text())
    try:
        result = asyncio.run(sample(config))
    except Exception as error:
        result = {"status": "agent_error", "error": repr(error)}
    write_json(Path(config["output_dir"]) / "result.json", result)


if __name__ == "__main__":
    main()
