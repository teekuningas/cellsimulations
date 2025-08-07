import numpy as np
from .common import State, Cell


class Rule:
    """Base class for rules."""

    def apply(self, state: State) -> State:
        """Apply rule to a state."""
        raise NotImplementedError


class Rule1D(Rule):
    """Implementation of a 1D elementary cellular automaton rule."""

    def __init__(self, rule_number: int):
        if not 0 <= rule_number <= 255:
            raise ValueError("Rule number must be between 0 and 255.")
        self.rule_number = rule_number
        # The rule number's binary representation defines the output for each of the 8 possible neighborhoods.
        # The neighborhoods are ordered from [1,1,1] down to [0,0,0].
        self.outputs = [
            Cell.ALIVE if (rule_number >> i) & 1 else Cell.DEAD for i in range(8)
        ]

    def apply_cell(self, neighborhood):
        """Applies the rule to a single 3-cell neighborhood."""
        # Convert neighborhood of Cell enums to integer indices (0-7)
        # [Cell.ALIVE, Cell.ALIVE, Cell.ALIVE] -> [1,1,1] -> 7
        # [Cell.DEAD, Cell.DEAD, Cell.DEAD] -> [0,0,0] -> 0
        index = (
            neighborhood[0].value * 4
            + neighborhood[1].value * 2
            + neighborhood[2].value * 1
        )
        return self.outputs[index]

    def apply(self, state: State) -> State:
        """Applies the rule to a 1D state."""
        if state.ndim != 1:
            raise ValueError("Rule1D can only be applied to 1D states.")

        data = state.data
        new_data = np.empty_like(data)

        for i in range(len(data)):
            # Get neighborhood with periodic boundary conditions
            left = data[i - 1]
            center = data[i]
            right = data[(i + 1) % len(data)]

            neighborhood = [left, center, right]
            new_data[i] = self.apply_cell(neighborhood)

        return State(new_data)


class RuleGameOfLife(Rule):
    """Implementation of Conway's Game of Life."""

    def apply(self, state: State) -> State:
        """Applies the Game of Life rule to a 2D state."""
        if state.ndim != 2:
            raise ValueError("RuleGameOfLife can only be applied to 2D states.")

        # Get the data as integer array (0 or 1)
        grid = np.array([[c.value for c in row] for row in state.data])

        # Pad the grid to handle periodic boundary conditions
        padded_grid = np.pad(grid, pad_width=1, mode="wrap")

        # Count live neighbors for each cell
        # This is done by summing the 3x3 window around each cell in the padded grid
        # and subtracting the cell's own value.
        neighbor_count = (
            padded_grid[:-2, :-2]  # top-left
            + padded_grid[:-2, 1:-1]  # top-center
            + padded_grid[:-2, 2:]  # top-right
            + padded_grid[1:-1, :-2]  # middle-left
            + padded_grid[1:-1, 2:]  # middle-right
            + padded_grid[2:, :-2]  # bottom-left
            + padded_grid[2:, 1:-1]  # bottom-center
            + padded_grid[2:, 2:]  # bottom-right
        )

        # Apply Game of Life rules:
        # 1. A living cell with 2 or 3 neighbors survives.
        # 2. A dead cell with 3 neighbors becomes alive.
        # All other cells die or stay dead.
        born = (grid == 0) & (neighbor_count == 3)
        survive = (grid == 1) & ((neighbor_count == 2) | (neighbor_count == 3))

        new_grid = np.full(grid.shape, Cell.DEAD)
        new_grid[born | survive] = Cell.ALIVE

        return State(new_grid)


class RuleBlock(Rule):
    """Implementation of a 2D block automaton rule."""

    def __init__(self, rules: dict):
        """Initializes the rule with a dictionary of patterns.
        The dictionary maps a tuple of 2x2 input cells to a tuple of 2x2 output cells.
        """
        self.rules = rules
        self.counter = 0

    def apply_block(self, block):
        """Applies the rule to a single 2x2 block."""
        # Convert block to a hashable tuple to use as a dictionary key
        block_tuple = tuple(block.flatten())
        return self.rules.get(block_tuple, block_tuple)

    def apply(self, state: State) -> State:
        """Applies the block rule to a 2D state."""
        if state.ndim != 2:
            raise ValueError("RuleBlock can only be applied to 2D states.")

        data = state.data
        n_rows, n_cols = data.shape
        new_data = np.copy(data)

        # Determine the phase (which 2x2 blocks to update)
        phase = self.counter % 2

        for r in range(phase, n_rows - 1, 2):
            for c in range(phase, n_cols - 1, 2):
                block = data[r : r + 2, c : c + 2]
                new_block_tuple = self.apply_block(block)
                new_data[r : r + 2, c : c + 2] = np.array(new_block_tuple).reshape(2, 2)

        self.counter += 1
        return State(new_data)


class RuleDiffusion(Rule):
    """Implementation of a diffusion rule (averaging over neighbors)."""

    def __init__(self, von_neumann=True, renormalize=True):
        """
        Initializes the diffusion rule.
        :param von_neumann: If True, use von Neumann neighborhood (4 neighbors).
                            If False, use Moore neighborhood (8 neighbors).
        :param renormalize: If True, renormalize the state to conserve the total sum.
        """
        self.von_neumann = von_neumann
        self.renormalize = renormalize

    def apply(self, state: State) -> State:
        """Applies the diffusion rule to a 2D state."""
        if state.ndim != 2:
            raise ValueError("RuleDiffusion can only be applied to 2D states.")

        grid = state.data.astype(float)

        # Kernel for convolution
        if self.von_neumann:
            kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        else:  # Moore
            kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

        # Use convolution to get the sum of the neighborhood
        from scipy.signal import convolve2d

        summed_neighbors = convolve2d(grid, kernel, mode="same", boundary="wrap")

        # The number of elements in the kernel is the divisor
        divisor = np.sum(kernel)

        new_grid = summed_neighbors / divisor

        if self.renormalize:
            previous_sum = np.sum(grid)
            new_sum = np.sum(new_grid)
            if new_sum != 0:
                new_grid = new_grid * (previous_sum / new_sum)

        return State(new_grid)
