import numpy as np
import matplotlib.pyplot as plt

from automaton.common import State, Cell
from automaton.rules import RuleGameOfLife
from automaton.visualization import animate_history_2d


def pad_pattern(pattern: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Pads a pattern to a specific shape, centering it."""
    padded = np.full(shape, Cell.DEAD)
    px, py = (shape[0] - pattern.shape[0]) // 2, (shape[1] - pattern.shape[1]) // 2
    padded[px : px + pattern.shape[0], py : py + pattern.shape[1]] = pattern
    return padded


if __name__ == "__main__":
    """Run as a script."""

    # Parameters
    n_generations = 200
    grid_size = (50, 50)
    rule_name = "Conway's Game of Life"

    # Create the rule
    rule = RuleGameOfLife()

    # Create the initial state with a glider
    glider_pattern = State.from_str(
        """
        .x.
        ..x
        xxx
    """,
        mapping={".": Cell.DEAD, "x": Cell.ALIVE},
    ).data

    initial_data = pad_pattern(glider_pattern, grid_size)
    state = State(initial_data)

    # Compute all generations
    history = [state]
    for _ in range(n_generations - 1):
        state = rule.apply(state)
        history.append(state)

    print(f"Computed {n_generations} generations of {rule_name}.")

    # Animate the history
    ani = animate_history_2d(history, rule_name=rule_name, interval=100)

    plt.show()
