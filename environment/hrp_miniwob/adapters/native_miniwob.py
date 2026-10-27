"""Frozen semantic-policy cache and atomic experiment JSON output."""

import hashlib
import json
import os
import sqlite3
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch



class FrozenDecoder:
    """Reference and reviewer share the immutable base, never the actor's LoRA."""

    def __init__(self, policy, path):
        self.policy = policy
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS probabilities (key TEXT PRIMARY KEY, payload TEXT, lp TEXT)")
        self.requests = self.forwards = self.prompt_tokens = 0

    def __call__(self, chat, codes):
        self.requests += 1
        payload = json.dumps({"messages": chat, "codes": codes}, ensure_ascii=False, sort_keys=True)
        key = hashlib.sha256(payload.encode()).hexdigest()
        cached = self.db.execute("SELECT lp FROM probabilities WHERE key=?", (key,)).fetchone()
        if cached:
            return np.asarray(json.loads(cached[0]))
        before = self.policy.forward_prompt_tokens
        disable = (self.policy.model.disable_adapter() if hasattr(self.policy.model, "disable_adapter")
                   else nullcontext())
        with disable, torch.no_grad():
            lp = self.policy.batch_log_probabilities([chat], [codes])[0].detach().cpu().numpy().copy()
        self.forwards += 1
        self.prompt_tokens += self.policy.forward_prompt_tokens - before
        self.db.execute("INSERT OR IGNORE INTO probabilities VALUES (?,?,?)", (key, payload, json.dumps(lp.tolist())))
        self.db.commit()
        recorded = np.asarray(json.loads(self.db.execute("SELECT lp FROM probabilities WHERE key=?", (key,)).fetchone()[0]))
        if np.max(np.abs(recorded - lp)) > 1e-3:
            raise AssertionError("Frozen reference replay differs across processes")
        return recorded


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
