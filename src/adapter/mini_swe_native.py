"""Official mini-SWE-agent loop and tool-call decoder in the task container."""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import yaml
from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.exceptions import FormatError
from minisweagent.models.litellm_model import LitellmModel

from src.benchmark.native_gateway import write_json
from src.hrp.native_measurement import fingerprint


def run(config: dict) -> dict:
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    source = Path(config["harness_root"]) / "src/minisweagent/config/mini.yaml"
    profile = yaml.safe_load(source.read_text())
    model_config = config["model"]
    model_kwargs = {"api_base": model_config["api_base"], "api_key": "EMPTY",
                    "max_tokens": model_config["max_tokens"], "seed": config["seed"]}
    if model_config.get("temperature") is not None:
        model_kwargs["temperature"] = model_config["temperature"]
    model_options = {**profile["model"], "model_name": "openai/"+model_config["id"].removeprefix("openai/"),
                     "model_kwargs": {**profile["model"]["model_kwargs"], **model_kwargs},
                     "cost_tracking": "ignore_errors"}
    model = LitellmModel(**model_options)
    if config["operation"] == "sample":
        state = json.loads(Path(config["state_path"]).read_text())
        # Frozen transport is enforced by the gateway; retain the original
        # parser/templates without executing an action or a controller prefix.
        saved = state["model_config"]
        model = LitellmModel(**{**saved, "model_name": model_options["model_name"],
            "model_kwargs": {**saved["model_kwargs"], **model_kwargs}})
        try:
            message = model.query(state["messages"])
            return {"status": "sampled", "message": message}
        except FormatError as error:
            return {"status": "format_error", "message": error.messages[0], "parser_error": str(error)}
    profile["agent"].pop("mode", None)  # Official unattended DefaultAgent; no CLI confirmation UI.
    agent = DefaultAgent(model, LocalEnvironment(**{**profile["environment"], "cwd": config["cwd"]}),
        **{**profile["agent"], "step_limit": config["budget"]["max_steps"] or 0, "cost_limit": 0,
           "wall_time_limit_seconds": math.ceil(config["budget"]["wall_time_sec"] or 0), "output_path": output / "trajectory.json"})
    original = model._query
    calls = 0
    previous_step = None
    previous_state = None

    def observed(messages, **kwargs):
        nonlocal calls, previous_step, previous_state
        calls += 1
        kind = "retry" if previous_step == agent.n_calls else "main"
        previous_step = agent.n_calls
        state_id = fingerprint([calls, messages, agent.n_calls])
        write_json(output / "states" / f"{state_id}.json", {
            "state_id": state_id, "harness": "mini-swe-agent", "call_kind": kind,
            "trace": {"agent_id": "root", "agent_role": "root", "parent_agent_id": None,
                      "previous_state_id": previous_state},
            "messages": messages, "model_config": model.config.model_dump(mode="json"),
            "controller": agent.serialize(), "n_calls": agent.n_calls,
            "consecutive_format_errors": agent.n_consecutive_format_errors,
            "elapsed_sec": time.time()-agent._start_time, "cwd": config["cwd"],
            "environment": agent.env.config.model_dump(mode="json"),
            "restore_scope": "frozen next model request; no running subprocess restoration"})
        previous_state = state_id
        return original(messages, **{**kwargs, "extra_headers": {
            **kwargs.get("extra_headers", {}), "X-HRP-State-ID": state_id, "X-HRP-Call-Kind": kind}})

    model._query = observed
    result = agent.run(config["instruction"])
    return {"status": result.get("exit_status", "unknown"), "native_result": result,
            "top_level_steps": agent.n_calls, "observed_provider_attempts": calls,
            "profile": "v2.4.6 mini.yaml + DefaultAgent unattended; dollar cap disabled; explicit experiment budget"}


def main():
    config = json.loads(Path(sys.argv[1]).read_text())
    try:
        result = run(config)
    except Exception as error:
        result = {"status": "agent_error", "error": repr(error)}
    write_json(Path(config["output_dir"]) / "result.json", result)


if __name__ == "__main__":
    main()
