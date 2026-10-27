"""Contract/control baseline: one public check before committing submission."""

MEMORY_STATES = 2
INITIAL_MEMORY = 1


def render(observation):
    return ("Compare the requested labels with the checked boxes. "
            "A rejected submission leaves the page available for correction.")


def review_mode(memory):
    return "program" if memory else "none"


def advance(memory, action, approved):
    return 0 if action == "submit" else memory
