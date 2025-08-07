import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import numpy as np

from .common import Cell, State


def plot_history_1d(history: list[State], rule_name: str = "1D Automaton"):
    """Makes a matplotlib plot of all generations of a 1D automaton."""
    fig, ax = plt.subplots()
    plt.axis("off")

    n_generations = len(history)
    n_cells = history[0].shape[0]

    ax.set_xlim([-0.5, n_cells - 0.5])
    ax.set_ylim([-0.5, n_generations - 0.5])
    ax.set_title(rule_name)
    ax.invert_yaxis()  # To have generation 0 at the top

    for gen_idx, state in enumerate(history):
        for cell_idx, cell in enumerate(state.data):
            if cell == Cell.ALIVE:
                facecolor = "black"
            else:
                facecolor = "white"

            rect = patches.Rectangle(
                (cell_idx - 0.5, gen_idx - 0.5),
                1,
                1,
                linewidth=0.5,
                edgecolor="green",
                facecolor=facecolor,
            )
            ax.add_patch(rect)
    return fig


def _get_color(cell):
    if cell == Cell.ALIVE:
        return "black"
    elif cell == Cell.DEAD:
        return "white"
    elif cell == Cell.WALL:
        return "grey"
    else:  # For continuous states
        val = max(0, min(1, float(cell)))
        return str(val)  # Grayscale value


def animate_history_2d(
    history: list[State], rule_name: str = "2D Automaton", interval=50
):
    """Makes a matplotlib animation from a history of 2D states."""
    fig, ax = plt.subplots()
    ax.axis("off")
    ax.set_title(rule_name)

    n_rows, n_cols = history[0].shape
    ax.set_xlim([-0.5, n_cols - 0.5])
    ax.set_ylim([-0.5, n_rows - 0.5])
    ax.invert_yaxis()

    # Create a grid of patches that will be updated
    patch_grid = []
    for r in range(n_rows):
        row_patches = []
        for c in range(n_cols):
            rect = patches.Rectangle(
                (c - 0.5, r - 0.5),
                1,
                1,
                linewidth=0.5,
                edgecolor="green",
                facecolor="white",
            )
            ax.add_patch(rect)
            row_patches.append(rect)
        patch_grid.append(row_patches)

    def _animation_frame(gen_idx):
        """Helper to generate a single frame."""
        state = history[gen_idx]
        ax.set_title(f"{rule_name} - Generation {gen_idx}")
        for r in range(n_rows):
            for c in range(n_cols):
                patch_grid[r][c].set_facecolor(_get_color(state.data[r, c]))

    ani = FuncAnimation(
        fig, _animation_frame, frames=len(history), interval=interval, blit=False
    )
    return ani
