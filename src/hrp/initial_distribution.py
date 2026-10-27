"""Discounted visits from an arbitrary start distribution on a finite DAG."""

import numpy as np


def discounted_visits(next_state, policy, initial, gamma):
    """Return sum_t gamma**t Pr(s_t=s), without nonterminal renormalization."""
    initial = np.asarray(initial, dtype=float)
    if (
        initial.shape != (len(next_state),)
        or np.any(initial < 0)
        or not np.isclose(initial.sum(), 1)
    ):
        raise ValueError(
            "Initial distribution must be a probability vector on graph rows"
        )
    if np.any(next_state >= np.arange(len(next_state))[:, None]) or np.any(
        next_state < -1
    ):
        raise ValueError("Finite graph rows must be topologically ordered")
    visits = initial.copy()
    for state in range(len(next_state) - 1, -1, -1):
        successors = next_state[state]
        live = successors >= 0
        np.add.at(visits, successors[live], gamma * visits[state] * policy[state, live])
    return visits
