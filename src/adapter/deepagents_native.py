"""Run the real DeepAgents graph inside the benchmark task container.

LocalShellBackend intentionally uses the container's actual filesystem. The
native file tools, explicit TodoList planning, general-purpose delegation and
summary middleware remain enabled. Callbacks do not replace the graph's loop.
"""
from __future__ import annotations

import asyncio
import contextvars
import importlib.metadata
import json
import os
import pickle
import sys
import time
from pathlib import Path

import httpx
from deepagents import create_deep_agent
from deepagents.backends.local_shell import LocalShellBackend
from langchain.agents.middleware import TodoListMiddleware
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.exceptions import ContextOverflowError
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.load import dumpd
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from openai import BadRequestError

from src.hrp.native_measurement import fingerprint

_current_call = contextvars.ContextVar("hrp_native_deepagents_call", default=None)


def serializable(value):
    return json.loads(json.dumps(value, default=lambda obj: dumpd(obj) if hasattr(obj, "to_json") else str(obj)))


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serializable(value), ensure_ascii=False, indent=2) + "\n")


class ExperimentBudgetExhausted(RuntimeError):
    pass


class OpenAIContextOverflowMiddleware(AgentMiddleware):
    """Translate OpenAI-compatible context 400s into LangChain's overflow signal.

    DeepAgents' native summarizer clips an oversized trailing tool result and
    retries only for ``ContextOverflowError``.  The pinned ChatOpenAI version
    exposes vLLM's equivalent response as ``BadRequestError`` instead.
    """

    async def awrap_model_call(self, request, handler):
        try:
            return await handler(request)
        except BadRequestError as error:
            message = str(error).lower()
            if "maximum context length" in message or "context length" in message and "tokens" in message:
                raise ContextOverflowError(str(error)) from error
            raise


class ForcedFirstActionMiddleware(AgentMiddleware):
    """Force one recorded native tool call before teacher continuation.

    This is an opt-in experiment hook.  It changes only the first model
    decision; subsequent decisions go through the configured model normally.
    The hook is used for the bounded one-step DP pilot and is deliberately
    recorded in the run manifest so its trajectories are not mistaken for
    ordinary teacher rollouts.
    """

    def __init__(self, action: dict) -> None:
        self.action = action
        self.used = False

    async def awrap_model_call(self, request: ModelRequest, handler) -> ModelResponse:
        if self.used:
            return await handler(request)
        self.used = True
        name = self.action["name"]
        args = self.action.get("args", {})
        return ModelResponse(result=[AIMessage(
            content="",
            tool_calls=[{
                "name": name,
                "args": args,
                "id": "forced-first-action",
                "type": "tool_call",
            }],
        )])


class Recorder(AsyncCallbackHandler):
    run_inline = True
    raise_error = True

    def __init__(self, config: dict, saver: InMemorySaver):
        self.config, self.saver = config, saver
        self.output = Path(config["output_dir"])
        self.output.mkdir(parents=True, exist_ok=True)
        self.started = time.time()
        self.model_calls = self.top_steps = self.output_tokens = 0
        self.http_attempts = 0
        self.budget_reason = None
        self.calls = {}
        self.tool_messages = set()
        self.parents = {}
        self.tool_runs = {}
        self.tool_decisions = {}
        self.previous_by_agent = {}

    def event(self, kind: str, **data) -> None:
        with (self.output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(serializable({"event": kind, "time": time.time(), **data}), ensure_ascii=False) + "\n")

    def exhaust(self, reason: str) -> None:
        self.budget_reason = reason
        self.event("budget_exhausted", reason=reason, top_steps=self.top_steps,
                   logical_model_calls=self.model_calls, http_attempts=self.http_attempts)
        raise ExperimentBudgetExhausted(f"Common experiment budget exhausted: {reason}")

    async def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, **kwargs):
        self.parents[str(run_id)] = str(parent_run_id) if parent_run_id is not None else None

    async def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None,
                                  tags=None, metadata=None, **kwargs):
        parameters = kwargs.get("invocation_params", {})
        metadata = metadata or {}
        namespace = metadata.get("langgraph_checkpoint_ns", "")
        kind = "summary" if metadata.get("lc_source") == "summarization" or not parameters.get("tools") else "subagent" if "tools:" in namespace else "decision"
        budget = self.config["budget"]
        if kind == "decision" and budget["max_steps"] is not None and self.top_steps >= budget["max_steps"]:
            self.exhaust("max_steps")
        if budget["max_model_calls"] is not None and self.http_attempts >= budget["max_model_calls"]:
            self.exhaust("max_model_calls")
        if budget["max_output_tokens"] is not None and self.output_tokens >= budget["max_output_tokens"]:
            self.exhaust("max_output_tokens")
        if budget["wall_time_sec"] is not None and time.time()-self.started >= budget["wall_time_sec"]:
            self.exhaust("wall_time_sec")
        self.model_calls += 1
        if kind == "decision":
            self.top_steps += 1
        state_id = fingerprint({"call": self.model_calls, "messages": serializable(messages),
                                "parameters": serializable(parameters), "metadata": metadata})
        parts = namespace.split("|")
        agent_id = "|".join(parts[:-1]) or "root"
        ancestor = str(parent_run_id) if parent_run_id is not None else None
        while ancestor and ancestor not in self.tool_runs:
            ancestor = self.parents.get(ancestor)
        parent_tool = self.tool_runs.get(ancestor, {})
        trace = {"agent_id": agent_id, "agent_role": "subagent" if "tools:" in namespace else "root",
                 "parent_agent_id": ("|".join(parts[:-2]) or "root") if "tools:" in namespace else None,
                 "parent_state_id": parent_tool.get("parent_state_id"),
                 "parent_tool_call_id": parent_tool.get("tool_call_id"),
                 "parent_tool_run_id": ancestor, "native_run_id": str(run_id),
                 "native_parent_run_id": str(parent_run_id) if parent_run_id is not None else None,
                 "previous_state_id": self.previous_by_agent.get(agent_id)}
        self.previous_by_agent[agent_id] = state_id
        checkpoint = self.output / "states" / (state_id + ".checkpoint.pkl")
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        # InMemorySaver.storage's defaultdict factory is a local lambda and
        # cannot be pickled. Persist its data, not the implementation factory.
        checkpoint.write_bytes(pickle.dumps({
            "storage": {thread: {ns: dict(entries) for ns, entries in spaces.items()}
                        for thread, spaces in self.saver.storage.items()},
            "writes": dict(self.saver.writes), "blobs": dict(self.saver.blobs)}))
        state = {"state_id": state_id, "harness": "deepagents", "call_kind": kind,
                 "trace": trace,
                 "messages": serializable(messages), "invocation_params": serializable(parameters),
                 "metadata": metadata, "tags": tags, "run_id": str(run_id),
                 "parent_run_id": str(parent_run_id), "cwd": os.getcwd(),
                 "relevant_environment": {k: os.environ[k] for k in
                    ("PATH", "LANG", "LC_ALL", "PYTHONPATH", "VIRTUAL_ENV") if k in os.environ},
                 "controller": {"checkpoint_file": str(checkpoint), "graph_metadata": metadata},
                 "budget": {"limits": budget, "top_level_steps": self.top_steps,
                            "logical_model_calls": self.model_calls, "output_tokens": self.output_tokens,
                            "http_attempts": self.http_attempts,
                            "elapsed_sec": time.time()-self.started},
                 "restore_scope": {"next_decision": "wire payload captured by HTTP hook",
                                   "full_inflight_controller": False,
                                   "limitations": ["Graph checkpoints omit live asyncio tasks, subprocesses and middleware stack frames."]}}
        self.calls[str(run_id)] = state
        _current_call.set(state)
        write(self.output / "states" / (state_id + ".json"), state)
        self.event("model_start", state_id=state_id, call_kind=kind, run_id=str(run_id), parent_run_id=str(parent_run_id))

    async def before_http(self, request: httpx.Request):
        state = _current_call.get()
        if state is None:
            self.event("unassociated_http_attempt", url=str(request.url).split("/v1/")[-1])
            raise RuntimeError("Model request has no native decision state")
        budget = self.config["budget"]
        if budget["max_model_calls"] is not None and self.http_attempts >= budget["max_model_calls"]:
            self.exhaust("max_model_calls")
        if budget["wall_time_sec"] is not None and time.time()-self.started >= budget["wall_time_sec"]:
            self.exhaust("wall_time_sec")
        self.http_attempts += 1
        state["http_attempts"] = state.get("http_attempts", 0) + 1
        request.headers["X-HRP-State-ID"] = state["state_id"]
        request.headers["X-HRP-Call-Kind"] = state["call_kind"] + ("_http_retry" if state["http_attempts"] > 1 else "")
        body = json.loads(request.content)
        state["wire_payload"] = body
        write(self.output / "states" / (state["state_id"] + ".json"), state)
        manifest_path = self.output / "manifest.json"
        if state["call_kind"] == "decision" and manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            manifest["tools"] = [tool["function"]["name"] for tool in body.get("tools", [])]
            manifest["tool_definitions"] = body.get("tools", [])
            write(manifest_path, manifest)
        self.event("http_attempt", state_id=state["state_id"], request=body,
                   attempt=state["http_attempts"], http_attempt=self.http_attempts)

    async def on_llm_end(self, response, *, run_id, **kwargs):
        generations = serializable(response.generations)
        outcomes = []
        state = self.calls.get(str(run_id), {})
        for group in response.generations:
            for generation in group:
                usage = getattr(generation.message, "usage_metadata", None) or {}
                self.output_tokens += usage.get("output_tokens", 0)
                message = generation.message
                for call in getattr(message, "tool_calls", []):
                    self.tool_decisions[(state.get("trace", {}).get("agent_id"), call.get("id"))] = state.get("state_id")
                finish_reason = (generation.generation_info or {}).get("finish_reason")
                outcomes.append({"finish_reason": finish_reason,
                    "decision_status": "length_truncated" if finish_reason == "length" else
                    "invalid_call" if getattr(message, "invalid_tool_calls", None) else
                    "tool_calls" if getattr(message, "tool_calls", None) else
                    "normal_end" if message.text.strip() else "empty_response"})
        self.event("model_end", state_id=state.get("state_id"), run_id=str(run_id),
                   generations=generations, llm_output=response.llm_output, outcomes=outcomes)

    async def on_llm_error(self, error, *, run_id, **kwargs):
        self.event("model_error", run_id=str(run_id), error=repr(error),
                   state_id=self.calls.get(str(run_id), {}).get("state_id"))

    async def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, **kwargs):
        namespace = (kwargs.get("metadata") or {}).get("langgraph_checkpoint_ns", "")
        agent_id = "|".join(namespace.split("|")[:-1]) or "root"
        tool_call_id = kwargs.get("tool_call_id")
        parent = self.tool_decisions.get((agent_id, tool_call_id))
        self.tool_runs[str(run_id)] = {"parent_state_id": parent, "tool_call_id": tool_call_id}
        self.event("tool_start", tool=serialized, input=input_str, run_id=str(run_id), parent_run_id=str(parent_run_id),
                   parent_state_id=parent, tool_call_id=tool_call_id)

    async def on_tool_end(self, output, *, run_id, **kwargs):
        self.event("tool_end", run_id=str(run_id), output=serializable(output))
        self.record_tool_messages(output)

    async def on_tool_error(self, error, *, run_id, **kwargs):
        self.event("tool_error", run_id=str(run_id), error=repr(error))

    def record_tool_messages(self, output) -> None:
        messages = output.get("messages", []) if isinstance(output, dict) else output
        messages = messages if isinstance(messages, (list, tuple)) else [messages]
        for message in messages:
            if isinstance(message, ToolMessage):
                key = fingerprint({"tool_call_id": message.tool_call_id, "name": message.name,
                                   "content": serializable(message.content), "status": message.status})
                if key not in self.tool_messages:
                    self.tool_messages.add(key)
                    self.event("tool_message", message=serializable(message))

    async def on_chain_end(self, outputs, **kwargs):
        # Tool argument validation can fail before BaseTool callbacks run. The
        # native ToolNode still emits the error as a ToolMessage to the graph.
        self.record_tool_messages(outputs)


async def run(config: dict) -> dict:
    os.chdir(config["cwd"])
    saver = InMemorySaver()
    recorder = Recorder(config, saver)
    model_config = config["model"]
    context_reserve_tokens = config.get("profile", {}).get("context_reserve_tokens", 1024)
    max_input_tokens = model_config["context_window"] - model_config["max_tokens"] - context_reserve_tokens
    if max_input_tokens <= 0:
        raise ValueError("DeepAgents context reserve leaves no input-token capacity")
    async with httpx.AsyncClient(timeout=1200, trust_env=False,
                                event_hooks={"request": [recorder.before_http]}) as client:
        model = ChatOpenAI(model=model_config["id"], base_url=model_config["api_base"],
                           api_key=model_config.get("api_key", "EMPTY"),
                           temperature=model_config.get("temperature"),
                           max_tokens=model_config["max_tokens"], seed=config["seed"],
                           use_responses_api=False, http_async_client=client,
                           profile={"max_input_tokens": max_input_tokens,
                                    "max_output_tokens": model_config["max_tokens"], "tool_calling": True})
        backend = LocalShellBackend(root_dir=config["cwd"], virtual_mode=False, inherit_env=True)
        middleware = [TodoListMiddleware(), OpenAIContextOverflowMiddleware()]
        forced_first_action = config.get("forced_first_action")
        if forced_first_action is not None:
            middleware.append(ForcedFirstActionMiddleware(forced_first_action))
        agent = create_deep_agent(model=model, backend=backend, checkpointer=saver,
                                  middleware=middleware)
        write(recorder.output / "manifest.json", {
            "harness": "deepagents", "version": importlib.metadata.version("deepagents"),
            "interface": "native", "profile": "create_deep_agent-LocalShellBackend-TodoList-v1",
            "tools": [], "tools_source": "actual first root model request; not inferred from names",
            "backend": {"type": "LocalShellBackend", "root_dir": config["cwd"], "virtual_mode": False, "inherit_env": True},
            "subagents": "default general-purpose; same model and backend", "permissions": "native allow; task-container isolation",
            "summary_and_tool_repair": "native defaults plus provider-compatible post-summary overflow-tail repair", "custom_skills": [], "custom_memory": [],
            "context_budget": {"context_window": model_config["context_window"],
                "max_output_tokens": model_config["max_tokens"],
                "summary_input_threshold": max_input_tokens,
                "reserve_tokens": context_reserve_tokens},
            "graph_recursion_guard": sys.maxsize if config["budget"]["max_model_calls"] is None else max(1000, config["budget"]["max_model_calls"]*10),
            "modifications": ["read-only observation callbacks", "common experiment budget", "explicit model sampling parameters",
                              "official TodoListMiddleware added explicitly; this is a fixed coding profile, not the unmodified default",
                              "OpenAI-compatible context-limit 400s mapped to ContextOverflowError so the native overflow clip-and-retry path runs",
                              "DeepAgents native overflow-tail repair also applied when a proactive summary still exceeds the provider's exact token limit",
                              "forced first native action" if forced_first_action is not None else ""],
            "restore_scope": "frozen next-model request; full in-flight graph is not restorable",
            "budget_enforcement": {"model_calls": "each HTTP attempt, including SDK retries",
                "wall_time": "async graph deadline; task-container supervisor must stop live subprocesses",
                "output_tokens": "checked between calls; in-flight calls may exceed remaining allowance"},
            "model": {k: v for k, v in model_config.items() if k not in ("api_key", "api_base")},
            "budget": config["budget"], "seed": config["seed"],
            "forced_first_action": forced_first_action})
        status, error, output = "completed", None, None
        try:
            limit = config["budget"]["wall_time_sec"]
            remaining = None if limit is None else max(0, limit - (time.time()-recorder.started))
            async with asyncio.timeout(remaining):
                output = await agent.ainvoke({"messages": [HumanMessage(content=config["instruction"])]},
                    config={"configurable": {"thread_id": "hrp-native"}, "callbacks": [recorder],
                            "recursion_limit": sys.maxsize if config["budget"]["max_model_calls"] is None else max(1000, config["budget"]["max_model_calls"]*10)})
        except ExperimentBudgetExhausted as exc:
            status, error = "budget_exhausted", str(exc)
        except TimeoutError:
            recorder.budget_reason = "wall_time_sec"
            recorder.event("budget_exhausted", reason="wall_time_sec")
            status, error = "budget_exhausted", "Common experiment wall-time budget exhausted"
        except Exception as exc:
            status, error = "budget_exhausted" if recorder.budget_reason else "agent_error", repr(exc)
        result = {"status": status, "error": error, "output": serializable(output),
                  "top_level_steps": recorder.top_steps, "logical_model_calls": recorder.model_calls,
                  "http_attempts": recorder.http_attempts, "budget_reason": recorder.budget_reason,
                  "output_tokens": recorder.output_tokens, "wall_time_sec": time.time()-recorder.started}
        write(recorder.output / "result.json", result)
        return result


def main():
    config = json.loads(Path(sys.argv[1]).read_text())
    asyncio.run(sample(config) if config.get("operation") == "sample" else run(config))


async def sample(config: dict) -> dict:
    """One frozen provider request decoded by the original LangChain adapter."""
    from openai import AsyncOpenAI
    state_path = Path(config["state_path"])
    original = state_path.read_bytes()
    state = json.loads(original)
    payload = dict(state["wire_payload"])
    model = config["model"]
    payload.update(model=model["id"], stream=False)
    payload.pop("stream_options", None)
    async with AsyncOpenAI(base_url=model["api_base"], api_key=model["api_key"],
                           timeout=config["budget"]["wall_time_sec"],
                           default_headers={"X-HRP-State-ID": state["state_id"],
                                            "X-HRP-Call-Kind": state["call_kind"]}) as client:
        response = await client.chat.completions.create(**payload)
    decoder = ChatOpenAI(model=model["id"], api_key="EMPTY", use_responses_api=False)
    message = decoder._create_chat_result(response).generations[0].message
    result = {"harness": "deepagents", "interface": "native", "operation": "sample",
              "status": "completed", "message": message.model_dump(mode="json"),
              "source_state_id": state["state_id"], "tools_executed": 0,
              "source_unchanged": state_path.read_bytes() == original,
              "restore_scope": "frozen next request decoded by native ChatOpenAI; no graph/tool execution"}
    write(Path(config["output_dir"]) / "result.json", result)
    return result


if __name__ == "__main__":
    main()
