"""Three-family native harness, frozen priors, and coverage Exact-PG.

The v1 loss/worker and decoder implementations are reused without changing
their source. This module owns every v2-specific observation and transition.
"""

import hashlib
import json
import string
from collections import defaultdict, deque
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from hrp_miniwob.browser import Step
from hrp_miniwob.lm_policy import LanguagePolicy as V1Policy
from hrp_miniwob.measurements import policy_metrics
from hrp_miniwob.oracle import evaluate, solve
from hrp_miniwob.suite import reachable_states
from hrp_miniwob.v2.browser import State, Task
from hrp_miniwob.v2.model import public_approve, state_category

from hrp_miniwob.adapters.meta_miniwob import Candidate as V1Candidate
from hrp_miniwob.adapters.meta_miniwob import MetaAdapter as V1Adapter
from hrp_miniwob.adapters.meta_miniwob import check_context
from hrp_miniwob.adapters.native_miniwob import write_json
from src.hrp.initial_distribution import discounted_visits

MICROBATCH_STATES = 8


def microbatches(policy, prompts, codes, size=MICROBATCH_STATES):
    """Group by menu size and a prompt-local length bucket, preserving row IDs."""
    if not 1 <= size <= MICROBATCH_STATES:
        raise ValueError("Microbatch must contain between one and eight useful states")
    groups = defaultdict(list)
    for index, (prompt, menu) in enumerate(zip(prompts, codes, strict=True)):
        length = len(policy.tokenizer(policy.chat_text(prompt), add_special_tokens=False,
                                      truncation=False)["input_ids"])
        if length > 2048:
            raise ValueError("Public prompt exceeds the preregistered context budget")
        # Always include padding, so attention-mask execution cannot depend on
        # whether another observation happens to fill the bucket exactly.
        groups[len(menu), (length // 64 + 1) * 64].append(index)
    for (_, length), indices in groups.items():
        for start in range(0, len(indices), size):
            yield indices[start:start + size], length


class LanguagePolicy(V1Policy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for code in string.ascii_uppercase:
            ids = self.tokenizer.encode(code, add_special_tokens=False)
            if len(ids) != 1 or self.tokenizer.decode(ids) != code:
                raise ValueError(f"v2 action code is not one reversible token: {code}")
            self.code_token_ids[code] = ids[0]
        if len(set(self.code_token_ids.values())) != len(self.code_token_ids):
            raise ValueError("v2 action tokens must be distinct")
        self.metadata["task_version"] = "v2"
        self.forward_microbatches = self.gradient_microbatches = 0
        self.metadata["decoder"].update(
            native_codes=string.ascii_uppercase, alias_codes=None,
            main_protocol_batch_size=MICROBATCH_STATES,
            larger_batches="split into fixed-shape microbatches of 8 states",
            on_policy_replay="fixed B=8 and prompt-local length bucket for rollout, reference, DP and gradients",
            optimizer_updates="sum microbatch losses; average over episodes or tasks only")
        self.metadata["prompt_encoding"].update(
            padding="strictly next multiple of 64 tokens; fixed eight rows; repeat last row to fill tails",
            padding_rows="excluded from returned probabilities, loss and useful-observation counters")
        self.metadata["counter_definition"].update(
            forward="useful state rows; excludes repeated padding rows",
            gradient="useful state rows in gradient-enabled forwards",
            forward_microbatches="actual B=8 model forwards, including padded tails",
            gradient_microbatches="actual B=8 gradient-enabled model forwards; replay checks included")

    def inputs(self, prompts, length):
        count = len(prompts)
        if not 1 <= count <= MICROBATCH_STATES:
            raise ValueError("A v2 forward needs between one and eight real states")
        chats = [self.chat_text(prompt) for prompt in prompts]
        chats += [chats[-1]] * (MICROBATCH_STATES - count)
        inputs = self.tokenizer(chats, padding="max_length", max_length=length,
                                truncation=False, add_special_tokens=False,
                                return_tensors="pt").to(self.device)
        inputs["position_ids"] = (inputs.attention_mask.cumsum(-1) - 1).clamp_min(0)
        return inputs

    def count_forward(self, inputs, count):
        tokens = int(inputs.attention_mask[:count].sum())
        self.forward_observations += count
        self.forward_prompt_tokens += tokens
        self.forward_microbatches += 1
        if torch.is_grad_enabled():
            self.gradient_observations += count
            self.gradient_prompt_tokens += tokens
            self.gradient_microbatches += 1

    def batch_log_probabilities(self, prompts, codes):
        if not prompts or len(prompts) != len(codes):
            raise ValueError("Provide one nonempty action menu for each prompt")
        if len({len(row) for row in codes}) != 1 or any(not row or len(set(row)) != len(row) for row in codes):
            raise ValueError("A differentiable batch needs equal-size, distinct action menus")
        result = [None] * len(prompts)
        for indices, length in microbatches(self, prompts, codes):
            inputs = self.inputs([prompts[i] for i in indices], length)
            logits = self.model(**inputs, use_cache=False, logits_to_keep=1).logits[:len(indices), -1, :]
            tokens = torch.tensor([[self.code_token_ids[code] for code in codes[i]] for i in indices],
                                  dtype=torch.long, device=self.device)
            lp = logits.gather(1, tokens).double().log_softmax(-1)
            self.count_forward(inputs, len(indices))
            for index, row in zip(indices, lp.unbind(), strict=True):
                result[index] = row
        return torch.stack(result)

    def batch_values(self, prompts, value_head):
        result = [None] * len(prompts)
        for indices, length in microbatches(self, prompts, [["value"]] * len(prompts)):
            inputs = self.inputs([prompts[i] for i in indices], length)
            hidden = self.model.get_base_model().model(**inputs, use_cache=False).last_hidden_state[:, -1, :]
            # Keep the value-head GEMM at B=8, including partial microbatches.
            values = value_head(hidden.float()).squeeze(-1).double()[:len(indices)]
            self.count_forward(inputs, len(indices))
            for index, value in zip(indices, values.unbind(), strict=True):
                result[index] = value
        return torch.stack(result)


class Candidate(V1Candidate):
    def messages(self, state):
        codes = list(string.ascii_uppercase[:len(state.task.actions)])
        observation = state.public()
        observation.update(action_check=self.review_mode(state.memory),
                           checked_actions=[a for a in state.task.actions if state.task.checked_action(a)],
                           actions=dict(zip(codes, state.task.actions, strict=True)))
        extra = self.render(json.loads(json.dumps(observation)))
        if not isinstance(extra, str) or extra != self.render(json.loads(json.dumps(observation))) or len(extra) > 4000:
            raise ValueError("Candidate must render deterministic text of at most 4000 characters")
        system = (
            "Operate the native MiniWoB page using one listed action code. Checkbox clicks toggle; "
            "each action request consumes one decision. Clicking a removed tile is a no-op. For ascending "
            "numbers, retain the initial visible layout after numbers disappear. Only exact native "
            "task completion earns 1; partial native credit is mapped to 0. All explicit costs "
            "are zero. Budget exhaustion terminates. Checked actions use the named fixed reviewer "
            "unless mode is none; rejection preserves the page for correction but consumes one "
            "decision. Program review uses public contents and the benchmark's public synonym "
            "vocabulary; frozen LM review may err. Optimize discounted reward gamma=0.95 with "
            "KL regularization tau=0.1 to the fixed reference. Return exactly one action code.\n" + extra
        )
        return [{"role": "system", "content": system},
                {"role": "user", "content": json.dumps(observation, ensure_ascii=False)}], codes


@dataclass
class Graph:
    task: Task
    horizon: int
    candidate: Candidate
    next_state: np.ndarray
    task_rewards: np.ndarray
    reference_table: np.ndarray | None = None

    @property
    def initial_state(self):
        return self.index(self.horizon, self.candidate.initial_memory, self.task.initial)

    def index(self, remaining, memory, physical):
        return ((remaining - 1) * self.candidate.memories + memory) * self.task.physical_states + physical

    def state(self, index):
        level, physical = divmod(index, self.task.physical_states)
        remaining, memory = divmod(level, self.candidate.memories)
        return State(self.task, physical, remaining + 1, memory)

    def reference(self):
        if self.reference_table is None:
            raise RuntimeError("Measure the independent reference before evaluation")
        return self.reference_table


class Adapter(V1Adapter):
    def __init__(self, policy, candidate, reference_policy, reviewer_policy, cache):
        # Probability caches from the previous B=1 numerical contract are stale.
        super().__init__(policy, candidate, reference_policy, reviewer_policy,
                         Path(cache) / "fixed_microbatch8")

    def measure(self, model):
        graph = self.graph(model)
        lp = np.log(graph.reference()).copy()
        bank = reachable_states(graph)
        prompts, menus = zip(*(self.messages(graph.state(int(i))) for i in bank), strict=True)
        with torch.no_grad():
            for indices, _ in microbatches(self.policy, prompts, menus):
                lp[bank[indices]] = self.policy.batch_log_probabilities(
                    [prompts[i] for i in indices], [menus[i] for i in indices]).detach().cpu().numpy()
        return graph, lp

    def review(self, task, physical, action, mode):
        if mode == "none":
            return True
        if mode == "program":
            return public_approve(task, physical, action)
        observation = State(task, physical, 1).public()
        # The fixed checker's decision deliberately has no budget/controller input.
        del observation["decisions_remaining"], observation["controller_memory"]
        observation["proposed_action"] = action
        if mode == "lm_tool":
            observation["program_approved"] = public_approve(task, physical, action)
        chat = [{"role": "system", "content": "Check the proposed action against the public instruction and current page. Approve only if it avoids an incorrect native terminal outcome. Use the remembered initial layout for hidden numbers. Answer A to approve or B to reject."},
                {"role": "user", "content": json.dumps(observation, ensure_ascii=False)}]
        check_context(self.reviewer.policy, chat)
        return bool(np.argmax(self.reviewer(chat, ["A", "B"])) == 0)

    def graph(self, model, with_reference=True):
        key, c = model.task.key, self.candidate
        if key not in self.graphs:
            shape = model.horizon * c.memories * model.task.physical_states, len(model.task.actions)
            graph = Graph(model.task, model.horizon, c, np.full(shape, -1, dtype=np.int64), np.zeros(shape))
            decisions = {}
            for index in range(shape[0]):
                state = graph.state(index)
                for a, action in enumerate(model.task.actions):
                    approved = True
                    if model.task.checked_action(action):
                        review_key = c.review_mode(state.memory), state.physical, action
                        if review_key not in decisions:
                            decisions[review_key] = self.review(model.task, state.physical, action, review_key[0])
                        approved = decisions[review_key]
                    memory = c.advance(state.memory, action, approved)
                    successor = int(model.physical_next[state.physical, a]) if approved else state.physical
                    if approved:
                        graph.task_rewards[index, a] = model.physical_rewards[state.physical, a]
                    if successor >= 0 and state.remaining > 1:
                        graph.next_state[index, a] = graph.index(state.remaining - 1, memory, successor)
            self.graphs[key] = graph
        graph = self.graphs[key]
        if with_reference and graph.reference_table is None:
            graph.reference_table = np.full(graph.next_state.shape, 1 / len(model.task.actions))
            bank = reachable_states(graph)
            prompts, menus = zip(*(self.messages(graph.state(int(i))) for i in bank), strict=True)
            # Populate the same immutable cache in batches; live rollout reads
            # these rows through the unchanged single-request cache interface.
            pending = {}
            for chat, codes in zip(prompts, menus, strict=True):
                payload = json.dumps({"messages": chat, "codes": codes}, ensure_ascii=False, sort_keys=True)
                key = hashlib.sha256(payload.encode()).hexdigest()
                if self.frozen.db.execute("SELECT 1 FROM probabilities WHERE key=?", (key,)).fetchone() is None:
                    pending[key] = (payload, chat, codes)
            keys, rows = list(pending), list(pending.values())
            frozen = self.frozen.policy
            disable = frozen.model.disable_adapter() if hasattr(frozen.model, "disable_adapter") else nullcontext()
            with disable, torch.no_grad():
                for indices, _ in microbatches(frozen, [r[1] for r in rows], [r[2] for r in rows]):
                    before = frozen.forward_prompt_tokens
                    lp = frozen.batch_log_probabilities([rows[i][1] for i in indices],
                                                        [rows[i][2] for i in indices]).detach().cpu().numpy()
                    self.frozen.forwards += len(indices)
                    self.frozen.prompt_tokens += frozen.forward_prompt_tokens - before
                    for i, row in zip(indices, lp, strict=True):
                        self.frozen.db.execute("INSERT OR IGNORE INTO probabilities VALUES (?,?,?)",
                                               (keys[i], rows[i][0], json.dumps(row.tolist())))
                        recorded = json.loads(self.frozen.db.execute("SELECT lp FROM probabilities WHERE key=?", (keys[i],)).fetchone()[0])
                        if np.max(np.abs(recorded - row)) > 1e-3:
                            raise AssertionError("Frozen reference replay differs across processes")
                    self.frozen.db.commit()
            for index, chat, codes in zip(bank, prompts, menus, strict=True):
                graph.reference_table[index] = np.exp(self.frozen(chat, codes))
        return graph


class Harness:
    def __init__(self, browser, adapter):
        self.browser, self.adapter = browser, adapter
        self.state = None
        self.counts = dict.fromkeys(("actor_decisions", "browser_actions", "program_checks", "reviewer_calls", "dom_events"), 0)
        self.last_review = None

    def reset(self, task):
        state = self.browser.reset(task.seed, task.family)
        if state.task != task:
            raise AssertionError("Public task changed on reset")
        self.state = replace(state, memory=self.adapter.candidate.initial_memory)
        return self.state

    def step(self, action):
        state, c = self.state, self.adapter.candidate
        if state is None or state.terminated:
            raise RuntimeError("An active task is required")
        if action not in state.task.actions:
            raise ValueError("Unknown action")
        self.counts["actor_decisions"] += 1
        approved, self.last_review = True, None
        if state.task.checked_action(action):
            mode = c.review_mode(state.memory)
            self.counts["program_checks"] += int(mode in ("program", "lm_tool"))
            self.counts["reviewer_calls"] += int(mode in ("lm", "lm_tool"))
            approved = self.adapter.review(state.task, state.physical, action, mode)
            self.last_review = {"mode": mode, "approved": approved}
        memory = c.advance(state.memory, action, approved)
        if not approved:
            remaining = state.remaining - 1
            if remaining == 0:
                self.browser.page.evaluate("core.endEpisode(0, false, 'decision budget exhausted')")
            self.state = replace(state, remaining=remaining, memory=memory, terminated=remaining == 0)
            self.browser.state = self.state
            return Step(self.state, 0., 0., 0. if remaining == 0 else None, "review_rejected")
        before = self.browser.dom_events
        result = self.browser.step(action)
        self.counts["browser_actions"] += 1
        self.counts["dom_events"] += self.browser.dom_events - before
        self.state = replace(result.state, memory=memory)
        self.browser.state = self.state
        return Step(self.state, result.reward, 0., result.raw_reward, result.reason)


def start_bank(graph):
    """50% native + 50% stratified legal starts, each with a real replay prefix.

    No optimal policy, reward-to-go or actor visitation chooses these starts.
    Coverage excludes the native row. The mixture gives it exactly 0.5 mass;
    every other reachable row receives positive mass, as in v1.
    """
    paths = {graph.initial_state: ()}
    queue = deque(paths)
    while queue:
        source = queue.popleft()
        for action, target in enumerate(graph.next_state[source]):
            target = int(target)
            if target >= 0 and target not in paths:
                paths[target] = (*paths[source], action)
                queue.append(target)
    groups = defaultdict(list)
    for index in sorted(paths):
        if index == graph.initial_state:
            continue
        state = graph.state(index)
        budget = (state.remaining - 1) // 3
        groups[state_category(graph.task, state.physical), budget, state.memory].append(index)
    nu = np.zeros(len(graph.next_state))
    for indices in groups.values():
        nu[indices] = 1 / len(groups) / len(indices)
    mu = .5 * nu
    mu[graph.initial_state] += .5
    if not np.array_equal(np.flatnonzero(mu), reachable_states(graph)):
        raise AssertionError("Coverage support differs from the fixed reachable bank")
    return mu, nu, paths


def replay_start(harness, graph, index, paths):
    state = harness.reset(graph.task)
    for action in paths[index]:
        state = harness.step(graph.task.actions[action]).state
        if state.terminated:
            raise AssertionError("Coverage prefix terminated before its chosen state")
    if state != graph.state(index):
        raise AssertionError("Real coverage prefix differs from the finite graph")
    return state


def exact_update(actor, model, adapter):
    from hrp_miniwob.adapters.miniwob_v2_batches import exact_batch_update

    return {**exact_batch_update(actor, [model], adapter),
            "task_seed": model.task.seed, "family": model.task.family}


def evaluate_models(models, adapter, output):
    output = Path(output)
    (output / "policies").mkdir(parents=True, exist_ok=True)
    rows = []
    for model in models:
        graph, lp = adapter.measure(model)
        policy = np.exp(lp)
        row = policy_metrics(graph, policy, np.zeros(policy.shape[1]), graph.reference())
        mu, _, _ = start_bank(graph)
        actual = evaluate(graph.next_state, graph.task_rewards, graph.reference(), policy, graph.initial_state, .95, .1)
        optimum = solve(graph.next_state, graph.task_rewards, graph.reference(), .95, .1)
        visits = discounted_visits(graph.next_state, policy, mu, .95)
        j, ceiling = float(mu @ actual.value), float(mu @ optimum.value)
        mismatch = float(.1 * visits @ np.sum(policy * (lp - optimum.log_policy), axis=1))
        row.update(mixed_J=j, mixed_ceiling=ceiling, mixed_mismatch=mismatch, mixed_gap=ceiling - j,
                   mixed_gap_identity_error=abs(ceiling - j - mismatch))
        if max(row["gap_identity_error"], row["mixed_gap_identity_error"]) > 1e-8:
            raise AssertionError("Independent KL gap identity failed")
        row.update(task_seed=model.task.seed, family=model.task.family, content_sha256=model.task.fingerprint)
        rows.append(row)
        np.savez_compressed(output / "policies" / f"{model.task.key}.npz", log_policy=lp,
                            reference=graph.reference(), next_state=graph.next_state, rewards=graph.task_rewards,
                            fixed_bank=reachable_states(graph), initial_distribution=mu)
        write_json(output / "progress.json", {"tasks": len(rows), "total": len(models)})
    if not rows:
        raise ValueError("Evaluation split is empty")
    keys = [key for key in rows[0] if key not in ("task_seed", "family", "content_sha256")]
    per_family = {f: {key: float(np.mean([r[key] for r in rows if r["family"] == f])) for key in keys}
                  for f in sorted({r["family"] for r in rows})}
    report = {"version": "v2", "tasks": len(rows), "aggregation": "equal task weight within family; equal family weight",
              "metrics": {key: float(np.mean([r[key] for r in per_family.values()])) for key in keys},
              "per_family": per_family, "rows": rows, "candidate_sha256": adapter.candidate.sha256,
              "backend": "exact finite graph; native DOM audits; independent frozen reference"}
    write_json(output / "metrics.json", report)
    return report
