import os
import numpy as np
import matplotlib.pyplot as plt

from automaton.common import State, Cell
from automaton.rules import Rule1D
from automaton.visualization import plot_history_1d

if __name__ == "__main__":
    """Run as a script."""

    # Parameters
    n_generations = 100
    n_cells = 101
    rule_number = 110  # Rule 110 is known to be Turing complete

    # Create the rule
    rule = Rule1D(rule_number)
    rule_name = f"Rule {rule_number}"

    # Create the initial state (a single 'ALIVE' cell in the middle)
    initial_data = np.full(n_cells, Cell.DEAD)
    initial_data[n_cells // 2] = Cell.ALIVE
    state = State(initial_data)

    # Compute all generations
    history = [state]
    for _ in range(n_generations - 1):
        state = rule.apply(state)
        history.append(state)

    print(f"Computed {n_generations} generations of {rule_name}.")

    # Plot the history
    fig = plot_history_1d(history, rule_name=rule_name)

    # Save the plot
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"rule_{rule_number}.pdf")
    fig.savefig(output_path)
    print(f"History plot saved to {output_path}")

    plt.show()
