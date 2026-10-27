"""CPU/native-browser acceptance; these diagnostics are not LM experiment results."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from hrp_miniwob.validation.check_miniwob_meta_harness import ProbePolicy
from hrp_miniwob.measurements import policy_gradient_weights
from hrp_miniwob.oracle import evaluate

from hrp_miniwob.adapters import miniwob_v2 as v2
from src.hrp.initial_distribution import discounted_visits


class TabularPolicy(ProbePolicy):
    def __init__(self):
        super().__init__()
        self.parameters = {}

    def batch_log_probabilities(self, chats, codes):
        rows = []
        for chat, menu in zip(chats, codes, strict=True):
            key = json.dumps(chat, sort_keys=True)
            if key not in self.parameters:
                self.parameters[key] = torch.nn.Parameter(torch.linspace(-.2, .2, len(menu), dtype=torch.float64))
            rows.append(self.parameters[key].log_softmax(-1))
        return torch.stack(rows)


def gradient_check(graph):
    rng = np.random.default_rng(42)
    logits = rng.normal(0, .3, graph.next_state.shape)
    direction = rng.normal(size=logits.shape)

    def probabilities(z):
        p = np.exp(z - z.max(axis=1, keepdims=True))
        return p / p.sum(axis=1, keepdims=True)

    mu, _, _ = v2.start_bank(graph)

    def objective(z):
        value = evaluate(graph.next_state, graph.task_rewards, graph.reference(), probabilities(z), graph.initial_state, .95, .1)
        return float(mu @ value.value)

    policy = probabilities(logits)
    _, advantage = policy_gradient_weights(graph, policy, np.zeros(policy.shape[1]), graph.reference())
    visits = discounted_visits(graph.next_state, policy, mu, .95)
    analytical = float(np.sum(visits[:, None] * policy * advantage * direction))
    epsilon = 1e-5
    finite_difference = (objective(logits + epsilon * direction) - objective(logits - epsilon * direction)) / (2 * epsilon)
    error = abs(analytical - finite_difference)
    assert error < 1e-8, error
    return error


def exact_update_check(model, bound):
    policy = TabularPolicy()
    old_policy, bound.policy = bound.policy, policy
    graph, before = bound.measure(model)
    mu, _, _ = v2.start_bank(graph)

    def objective(lp):
        return float(mu @ evaluate(graph.next_state, graph.task_rewards, graph.reference(), np.exp(lp), graph.initial_state, .95, .1).value)

    optimizer = torch.optim.SGD(list(policy.parameters.values()), lr=.1)
    engine = SimpleNamespace(policy=policy, optimizer_zero_grad=lambda: optimizer.zero_grad(set_to_none=True), lr_scheduler_step=lambda: None)

    def step():
        norm = torch.nn.utils.clip_grad_norm_(list(policy.parameters.values()), 1.)
        optimizer.step()
        return norm

    engine.optimizer_step = step
    record = v2.exact_update(SimpleNamespace(engine=engine), model, bound)
    _, after = bound.measure(model)
    delta = objective(after) - objective(before)
    assert delta > 0, delta
    bound.policy = old_policy
    return {"mixed_J_increase_one_tabular_update": delta, **record}
