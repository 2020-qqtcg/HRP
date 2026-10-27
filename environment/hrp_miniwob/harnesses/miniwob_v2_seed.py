"""v2 baseline: public observation, direct execution, no review."""

MEMORY_STATES = 1
INITIAL_MEMORY = 0


def render(observation):
    return "Follow the instruction using the listed controls. Check your current selections before committing."


def review_mode(memory):
    return "none"


def advance(memory, action, approved):
    return 0
