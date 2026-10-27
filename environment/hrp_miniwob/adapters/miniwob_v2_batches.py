"""Eight-state BF16 microbatches with unchanged task/episode loss weights."""

import numpy as np
import torch
from hrp_miniwob.measurements import policy_gradient_weights
from hrp_miniwob.suite import reachable_states

from hrp_miniwob.adapters.miniwob_v2 import MICROBATCH_STATES, microbatches, start_bank
from src.hrp.initial_distribution import discounted_visits


def exact_batch_update(actor, models, adapter, microbatch_size=MICROBATCH_STATES):
    """Average task gradients at one fixed actor, then take ONE optimizer step.

    Tasks are sampled uniformly by family, then instance by the caller. Each
    task retains its original coverage distribution and exact action expectation.
    Accumulate task means, not a mean over the pooled state rows.
    """
    if not models:
        raise ValueError("Exact-PG needs a nonempty task batch")
    actor.engine.optimizer_zero_grad()
    error, states, minimum = 0., 0, 1.
    rows = []
    for model in models:
        graph, lp = adapter.measure(model)
        policy = np.exp(lp)
        _, advantage = policy_gradient_weights(graph, policy, np.zeros(policy.shape[1]), graph.reference())
        mu, _, _ = start_bank(graph)
        weights = discounted_visits(graph.next_state, policy, mu, .95)
        bank = reachable_states(graph)
        states += len(bank)
        minimum = min(minimum, float(mu[bank].min()))
        for index in bank:
            chat, codes = adapter.messages(graph.state(int(index)))
            rows.append({"prompt": chat, "codes": codes, "log_policy": lp[index],
                         "advantage": advantage[index], "weight": weights[index] / len(models)})
    batches = 0
    for indices, _ in microbatches(actor.engine.policy, [r["prompt"] for r in rows],
                                   [r["codes"] for r in rows], microbatch_size):
        batch = [rows[i] for i in indices]
        current = actor.engine.policy.batch_log_probabilities(
            [r["prompt"] for r in batch], [r["codes"] for r in batch])
        error = max(error, float(np.max(np.abs(current.detach().cpu().numpy()
                                               - np.asarray([r["log_policy"] for r in batch])))))
        advantage = current.new_tensor(np.asarray([r["advantage"] for r in batch]))
        weights = current.new_tensor([r["weight"] for r in batch])
        loss = -(weights * (current.exp() * advantage).sum(-1)).sum()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite batched Exact-PG loss")
        loss.backward()
        batches += 1
        del current, loss
    if error > 1e-3:
        raise AssertionError(f"Policy replay failed: {error}")
    norm = actor.engine.optimizer_step()
    actor.engine.lr_scheduler_step()
    return {"task_batch_size": len(models), "task_seeds": [m.task.seed for m in models],
            "families": [m.task.family for m in models], "states": states,
            "grad_norm": float(norm), "replay_max_log_error": error,
            "actor_backward_observations": states, "actor_backward_microbatches": batches,
            "forward_microbatch_states": microbatch_size, "minimum_start_mass": minimum}
