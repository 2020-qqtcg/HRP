"""Reversible presentation variants for the frozen-policy readout study.

The finite task graph always uses native H0 semantic-action order.  A renderer
returns one code for each semantic column, so decoding the selected code maps
back to exactly that column even when the visible code assignment changes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hrp_miniwob.harnesses.miniwob_harnesses import CODES, Candidate


RENDERINGS = tuple(f"R{i}" for i in range(6))
RENDERING_SEED = 4204
H0_CANDIDATE = Path(__file__).resolve().parents[1] / "harnesses/miniwob_h0.py"


def semantic_to_code_indices(size: int, rendering_id: str) -> tuple[int, ...]:
    """Return the canonical code index assigned to every semantic action."""
    if rendering_id not in RENDERINGS:
        raise ValueError(f"Unknown readout rendering: {rendering_id}")
    if not 1 <= size <= len(CODES):
        raise ValueError("The semantic menu does not fit the one-token alphabet")
    if rendering_id in {"R0", "R1", "R2"}:
        order = np.arange(size)
    elif rendering_id == "R3":
        order = np.roll(np.arange(size), -1)
    elif rendering_id == "R4":
        order = np.arange(size)[::-1]
    else:
        # Reinitialize from the fixed seed for each menu size.  Consequently a
        # size has one preregistered mapping independent of task and state.
        order = np.random.default_rng(RENDERING_SEED).permutation(size)
    return tuple(int(i) for i in order)


def mapping_record(size: int, rendering_id: str) -> dict:
    permutation = semantic_to_code_indices(size, rendering_id)
    inverse = tuple(int(i) for i in np.argsort(permutation))
    return {
        "rendering_id": rendering_id,
        "menu_size": size,
        "canonical_codes": list(CODES[:size]),
        "semantic_to_code_index": list(permutation),
        "code_index_to_semantic": list(inverse),
        "fixed_points": sum(i == value for i, value in enumerate(permutation)),
        "rendering_seed": RENDERING_SEED if rendering_id == "R5" else None,
    }


@dataclass(frozen=True)
class RenderedPrompt:
    messages: list[dict[str, str]]
    semantic_codes: list[str]
    prompt_sha256: str
    mapping: dict


class ReadoutRenderer:
    """Render R0--R5 while preserving H0 state, actions, and execution."""

    def __init__(self, rendering_id: str):
        if rendering_id not in RENDERINGS:
            raise ValueError(f"Unknown readout rendering: {rendering_id}")
        self.rendering_id = rendering_id
        self.candidate = Candidate(H0_CANDIDATE)
        if self.candidate.profile != "H0":
            raise AssertionError("The readout baseline must remain current H0")

    def render(self, state) -> RenderedPrompt:
        canonical, canonical_codes = self.candidate.messages(state)
        observation = json.loads(canonical[1]["content"])
        actions = list(state.task.actions)
        if list(observation["actions"].values()) != actions:
            raise AssertionError("Canonical H0 menu lost semantic action order")
        mapping = mapping_record(len(actions), self.rendering_id)
        permutation = mapping["semantic_to_code_index"]
        semantic_codes = [canonical_codes[i] for i in permutation]

        if self.rendering_id == "R1":
            # JSON preserves insertion order.  The nested action dictionary is
            # deliberately left untouched.
            observation = dict(reversed(list(observation.items())))
        elif self.rendering_id == "R2":
            observation["actions"] = [
                {"code": code, "action": action}
                for code, action in zip(canonical_codes, actions, strict=True)
            ]
        elif self.rendering_id in {"R3", "R4", "R5"}:
            observation["actions"] = dict(zip(semantic_codes, actions, strict=True))

        messages = [canonical[0], {
            "role": "user",
            "content": json.dumps(observation, ensure_ascii=False),
        }]
        payload = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        return RenderedPrompt(
            messages=messages,
            semantic_codes=semantic_codes,
            prompt_sha256=hashlib.sha256(payload.encode()).hexdigest(),
            mapping=mapping,
        )


def decode_code(state, rendering_id: str, code: str) -> str:
    rendered = ReadoutRenderer(rendering_id).render(state)
    if code not in rendered.semantic_codes:
        raise ValueError("Code is outside the complete deployed menu")
    return state.task.actions[rendered.semantic_codes.index(code)]


def encoded_column_order(size: int, rendering_id: str) -> np.ndarray:
    """Semantic columns in canonical-code order for an independent solve."""
    permutation = semantic_to_code_indices(size, rendering_id)
    return np.argsort(np.asarray(permutation, dtype=np.int64))


def align_encoded_columns(values: np.ndarray, rendering_id: str) -> np.ndarray:
    """Map a canonical-code-ordered table back to semantic-action order."""
    permutation = semantic_to_code_indices(values.shape[1], rendering_id)
    return np.asarray(values)[:, np.asarray(permutation, dtype=np.int64)]
