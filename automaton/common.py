from enum import Enum
import numpy as np


class Cell(Enum):
    """Cells can be of three types."""

    ALIVE = 1
    DEAD = 0
    WALL = -1

    def __add__(self, other):
        if isinstance(other, Cell):
            return self.value + other.value
        else:
            return self.value + other

    def __radd__(self, other):
        if isinstance(other, Cell):
            return self.value + other.value
        else:
            return self.value + other


class State:
    """Stores a state."""

    def __init__(self, data):
        """Initialises a state from a numpy array."""
        self.data = data

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @classmethod
    def from_str(cls, s, mapping={"x": Cell.ALIVE, "o": Cell.DEAD, "w": Cell.WALL}):
        """Creates a state from a multiline string representation."""
        rows = []
        for row_str in s.strip().split("\n"):
            row = [mapping[char] for char in row_str.strip()]
            rows.append(row)
        return cls(np.array(rows, dtype=Cell))
