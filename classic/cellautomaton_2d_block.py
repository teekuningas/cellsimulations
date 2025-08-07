import matplotlib.pyplot as plt

from automaton.common import State, Cell
from automaton.rules import RuleBlock
from automaton.visualization import animate_history_2d


def get_gas_rules():
    """Returns the rule dictionary for the HPP gas model."""
    A, D, W = Cell.ALIVE, Cell.DEAD, Cell.WALL

    # Rules are defined as (in_tuple) -> out_tuple
    # The tuples are flattened 2x2 blocks.
    rules = {
        # Head-on collision
        (A, D, D, A): (D, A, A, D),
        (D, A, A, D): (A, D, D, A),
        # Wall collisions
        (D, A, W, W): (A, D, W, W),
        (A, D, W, W): (D, A, W, W),
        (W, A, W, D): (W, D, W, A),
        (W, D, W, A): (W, A, W, D),
        (W, W, A, D): (W, W, D, A),
        (W, W, D, A): (W, W, A, D),
        (A, W, D, W): (D, W, A, W),
        (D, W, A, W): (A, W, D, W),
    }
    return rules


if __name__ == "__main__":
    """Run as a script."""

    # Parameters
    n_generations = 200
    rule_name = "HPP Gas Model"

    # Create the rule
    gas_rules = get_gas_rules()
    rule = RuleBlock(gas_rules)

    # Create the initial state from a string
    initial_pattern_str = """
        wwwwwwwwwwwwwwwwwwwwwwww
        woooooooooooooooooooooow
        woooooooooooooooooxoooow
        wooxoooooooxooooooooooow
        woooooooooooooooooooooow
        woooooooxoooooooooooooow
        woooooooooooooooooooooow
        wooooxooooooooxoooooooow
        woooooooooooooooooooooow
        woooooooooooooooooxoooow
        woooooooooooooooooooooow
        wooooooooooxooooooooooow
        woooooooooooooooooxoooow
        woooooooxoooooooooooooow
        wooooxooooooooooooooooow
        woooooooooooooooxoooooow
        woooooooooooooooooooooow
        wwwwwwwwwwwwwwwwwwwwwwww
    """
    state = State.from_str(initial_pattern_str)

    # Compute all generations
    history = [state]
    for _ in range(n_generations - 1):
        state = rule.apply(state)
        history.append(state)

    print(f"Computed {n_generations} generations of {rule_name}.")

    # Animate the history
    ani = animate_history_2d(history, rule_name=rule_name, interval=100)

    plt.show()
