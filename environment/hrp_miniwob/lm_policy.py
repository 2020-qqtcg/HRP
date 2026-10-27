"""A Qwen3/Qwen3.5 LoRA policy with one token per complete semantic action.

The decoder conditions on the listed action tokens. It never generates free
text, reasoning tokens, or EOS, so validity is one by construction. Probabilities
are always returned in Task.actions order, including for reordered interfaces.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .browser import State
from .model import TaskModel

if TYPE_CHECKING:
    import torch


RULES = (
    "Each click toggles one checkbox. Every action uses one decision. "
    "Submit ends the task: reward 1 for exactly the requested selection, else 0. "
    "Exhausting the decisions without submitting ends the task with reward 0. "
    "All other task rewards are 0. Maximize expected sum over t of "
    "gamma^t * (reward - cost - tau * log(policy/reference)). "
    "Reply with exactly one listed action code."
)


def render_prompt(model: TaskModel, state_index: int, costs: np.ndarray,
                  reference: np.ndarray, style: str = "native",
                  gamma: float = 0.95, tau: float = 0.1) -> tuple[str, list[str]]:
    """Render public observations only; never read rewards, goals, or DP values.

    ``codes[a]`` is the token code for semantic action ``model.task.actions[a]``.
    Reordering changes presentation order, not that code-to-action mapping.
    Costs/reference can be an action vector or a full state/action table.
    """
    costs = np.asarray(costs)
    reference = np.asarray(reference)
    return render_state(
        model.state(state_index), costs if costs.ndim == 1 else costs[state_index],
        reference if reference.ndim == 1 else reference[state_index], style, gamma, tau,
    )


def render_state(state: State, costs: np.ndarray, reference: np.ndarray,
                 style: str = "native", gamma: float = 0.95,
                 tau: float = 0.1) -> tuple[str, list[str]]:
    """Render a live public observation without requiring a compiled task graph."""
    if style not in {"native", "json", "reordered", "aliases"}:
        raise ValueError(f"Unknown interface style: {style}")
    actions = state.task.actions
    alphabet = "0123456" if style == "aliases" else "ABCDEFG"
    codes = list(alphabet[:len(actions)])
    costs = np.asarray(costs)
    reference = np.asarray(reference)
    action_costs = costs
    action_reference = reference
    checkboxes = [
        {"id": f"ch{i}", "label": label, "checked": bool(state.mask & (1 << i))}
        for i, label in enumerate(state.task.labels)
    ]
    menu = [
        {"code": code, "action": action, "cost": float(cost),
         "reference": float(prior)}
        for code, action, cost, prior in zip(
            codes, actions, action_costs, action_reference, strict=True
        )
    ]
    if style == "json":
        payload = {
            "instruction": state.task.instruction,
            "decisions_remaining": state.remaining,
            "gamma": gamma,
            "tau": tau,
            "checkboxes": checkboxes,
            "actions": menu,
        }
        return RULES + "\n" + json.dumps(payload, ensure_ascii=False), codes
    if style == "reordered":
        checkboxes.reverse()
        menu.reverse()
    rows = [
        RULES,
        f"Instruction: {state.task.instruction}",
        f"Decisions remaining: {state.remaining}; gamma={gamma}; tau={tau}",
        "Checkboxes:",
    ]
    rows += [
        f'{box["id"]}: label={json.dumps(box["label"])}; checked={box["checked"]}'
        for box in checkboxes
    ]
    rows.append("Actions:")
    rows += [
        f'{item["code"]}: {item["action"]}; cost={item["cost"]}; '
        f'reference={item["reference"]}'
        for item in menu
    ]
    return "\n".join(rows), codes


class LanguagePolicy:
    """Qwen3/Qwen3.5 text LM, non-thinking chat, temperature-one decoder.

    The pretrained base is frozen. LoRA initialization has zero output, so
    changing ``seed`` changes adapter initialization without changing the initial
    policy. ``trainable=False`` with no adapter measures the unadapted base.
    The main protocol uses BF16 with one observation per forward, so each
    policy row is independent of other observations' padding and batch shape.
    Gradients accumulate over a task before its optimizer update. Larger batches
    remain available for numerical controls. Categorical normalization is float64.
    """

    def __init__(self, model_path: str = "Qwen/Qwen3-1.7B", seed: int = 0,
                 adapter_path: str | None = None, trainable: bool = True,
                 device: str = "cuda", rank: int = 8):
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        config = AutoConfig.from_pretrained(model_path)
        source_model_type = config.model_type
        if source_model_type not in {"qwen3", "qwen3_5", "qwen3_5_text"}:
            raise ValueError(
                f"Unsupported semantic policy model_type={source_model_type!r}"
            )
        config = config.get_text_config()
        target_modules = ["q_proj", "v_proj"] if source_model_type == "qwen3" else [
            "q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z",
            "in_proj_b", "in_proj_a", "out_proj", "gate_proj", "up_proj", "down_proj",
        ]
        config.use_cache = False
        config.attention_dropout = 0.0
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.code_token_ids = {}
        for code in "ABCDEFG0123456":
            encoded = self.tokenizer.encode(code, add_special_tokens=False)
            if len(encoded) != 1 or self.tokenizer.decode(encoded) != code:
                raise ValueError(f"Action code {code!r} is not exactly one reversible token")
            self.code_token_ids[code] = encoded[0]
        if len(set(self.code_token_ids.values())) != len(self.code_token_ids):
            raise ValueError("Action codes must have distinct token IDs")
        base = AutoModelForCausalLM.from_pretrained(
            model_path, config=config, dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to(self.device)
        base.requires_grad_(False)
        torch.manual_seed(seed)
        if adapter_path is not None:
            self.model = PeftModel.from_pretrained(
                base, adapter_path, is_trainable=trainable
            )
        elif trainable:
            self.model = get_peft_model(base, LoraConfig(
                task_type="CAUSAL_LM", r=rank, lora_alpha=2 * rank,
                lora_dropout=0.0, target_modules=target_modules,
                bias="none",
            ))
        else:
            self.model = base
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        self.model.train(trainable)
        self.forward_observations = 0
        self.forward_prompt_tokens = 0
        self.gradient_observations = 0
        self.gradient_prompt_tokens = 0
        self.parameter_counts = {
            "total": sum(p.numel() for p in self.model.parameters()),
            "trainable": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        }
        self.metadata = {
            "base_model": model_path, "adapter_path": adapter_path,
            "seed": seed, "rank": rank, "lora_alpha": 2 * rank,
            "source_model_type": source_model_type,
            "loaded_model_type": self.model.config.model_type,
            "target_modules": target_modules, "dropout": 0.0,
            "dtype": "bfloat16", "attention": "sdpa",
            "float32_matmul_precision": "highest", "allow_tf32": False,
            "decoder_temperature": 1.0, "enable_thinking": False,
            "code_token_ids": self.code_token_ids,
            "parameter_counts": self.parameter_counts,
            "decoder": {
                "kind": "one_token_conditional_categorical",
                "semantic_order": "Task.actions",
                "native_codes": "ABCDEFG", "alias_codes": "0123456",
                "temperature": 1.0, "log_softmax_dtype": "float64",
                "logits_dtype": "bfloat16", "logits_to_keep": 1,
                "generation_api_used": False, "eos_is_action": False,
                "use_cache": False,
                "on_policy_replay": "identical observations, ordering, and batch boundaries",
                "main_protocol_batch_size": 1,
                "optimizer_updates": "accumulate gradients over all task states before updating",
                "larger_batches": "numerical controls only in the main experiment protocol",
            },
            "prompt_encoding": {
                "messages": "explicit role messages; legacy strings become one user message",
                "chat_template": self.tokenizer.get_chat_template(),
                "chat_template_sha256": hashlib.sha256(
                    self.tokenizer.get_chat_template().encode()
                ).hexdigest(),
                "tokenize": False, "add_generation_prompt": True,
                "enable_thinking": False, "add_special_tokens": False,
                "padding_side": "left", "padding": True, "truncation": False,
                "pad_token_id": self.tokenizer.pad_token_id,
                "position_ids": "attention_mask.cumsum(-1)-1, clamped at zero",
            },
            "counter_definition": {
                "forward": "completed policy forwards, including gradient-enabled forwards",
                "gradient": "completed policy forwards while torch.is_grad_enabled()",
                "prompt_tokens": "sum of attention_mask after chat templating; excludes padding",
            },
        }

    def chat_text(self, prompt):
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def batch_log_probabilities(self, prompts: list[str],
                                codes: list[list[str]]) -> torch.Tensor:
        """Differentiable [batch, actions] log probabilities for equal-size menus."""
        import torch

        if len(prompts) != len(codes) or not prompts:
            raise ValueError("Provide one nonempty action menu for each prompt")
        if len({len(row) for row in codes}) != 1 or not codes[0]:
            raise ValueError("A differentiable batch must have the same positive action count")
        if any(len(set(row)) != len(row) for row in codes):
            raise ValueError("Each semantic action must have a distinct code")
        chats = [self.chat_text(prompt) for prompt in prompts]
        inputs = self.tokenizer(
            chats, padding=True, truncation=False, add_special_tokens=False,
            return_tensors="pt",
        )
        prompt_tokens = int(inputs["attention_mask"].sum())
        inputs = inputs.to(self.device)
        # Left padding puts every final real token at index -1. Explicit positions
        # also keep an observation's position IDs invariant to batch composition.
        position_ids = (inputs["attention_mask"].cumsum(dim=1) - 1).clamp_min(0)
        logits = self.model(
            **inputs, position_ids=position_ids, use_cache=False, logits_to_keep=1,
        ).logits[:, -1, :]
        self.forward_observations += len(prompts)
        self.forward_prompt_tokens += prompt_tokens
        if torch.is_grad_enabled():
            self.gradient_observations += len(prompts)
            self.gradient_prompt_tokens += prompt_tokens
        token_ids = torch.tensor(
            [[self.code_token_ids[code] for code in row] for row in codes],
            dtype=torch.long, device=self.device,
        )
        action_logits = logits.gather(1, token_ids).to(torch.float64)
        return torch.log_softmax(action_logits, dim=-1)

    def log_probabilities(self, prompts: list[str], codes: list[list[str]],
                          batch_size: int = 1) -> list[np.ndarray]:
        """Measure without gradients; preserve input order across variable menus."""
        import torch

        if len(prompts) != len(codes) or batch_size < 1:
            raise ValueError("Mismatched prompts/menus or nonpositive batch size")
        result = [np.empty(0) for _ in prompts]
        training = self.model.training
        self.model.eval()
        with torch.inference_mode():
            for action_count in sorted({len(row) for row in codes}):
                indices = [i for i, row in enumerate(codes) if len(row) == action_count]
                for start in range(0, len(indices), batch_size):
                    batch = indices[start:start + batch_size]
                    log_policy = self.batch_log_probabilities(
                        [prompts[i] for i in batch], [codes[i] for i in batch],
                    ).cpu().numpy()
                    for index, row in zip(batch, log_policy, strict=True):
                        result[index] = row
        self.model.train(training)
        return result

    def save_adapter(self, path: str | Path) -> None:
        """Save the trained adapter, tokenizer, and the finite decoder contract."""
        from peft import PeftModel

        if not isinstance(self.model, PeftModel):
            raise ValueError("The unadapted base has no LoRA adapter to save")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        (path / "policy_metadata.json").write_text(
            json.dumps(self.metadata, indent=2) + "\n"
        )
