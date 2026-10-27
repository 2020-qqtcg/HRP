"""Initial model-driven harness for the MiniWoB Meta-Harness search."""

MEMORY_STATES = 1
INITIAL_MEMORY = 0


def render(observation):
    return "Choose the next action to complete the user's checkbox task."


def review_mode(memory):
    return "none"


def advance(memory, action, approved):
    return 0
