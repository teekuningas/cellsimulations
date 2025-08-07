import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.collections as clt
from matplotlib.animation import FuncAnimation

import sys

sys.path.insert(0, ".")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.collections as clt
from matplotlib.animation import FuncAnimation

import numpy as np
import random

from automaton.common import State
from automaton.rules import RuleDiffusion


def init_plot(ax, sphere_radius, title):
    """Inits a single spherical subplot."""

    ax.axis("off")

    # Prepare the permanent plot properties
    ax.set_xlim([-2 * sphere_radius, 2 * sphere_radius])
    ax.set_ylim([-2 * sphere_radius, 2 * sphere_radius])

    ax.set_title(title)

    # Add a blue sphere
    sphere = plt.Circle((0, 0), sphere_radius, color="b")
    ax.add_patch(sphere)

    # Add a couple of green spheres
    sphere1 = plt.Circle(
        ((1 / 3) * sphere_radius, (1 / 3) * sphere_radius), sphere_radius / 4, color="g"
    )
    sphere2 = plt.Circle((-(1 / 3) * sphere_radius, 0), sphere_radius / 4, color="g")
    sphere3 = plt.Circle(
        ((1 / 3) * sphere_radius, -(1 / 3) * sphere_radius),
        sphere_radius / 4,
        color="g",
    )
    ax.add_patch(sphere1)
    ax.add_patch(sphere2)
    ax.add_patch(sphere3)

    return ax


def init_face_collection(n_layers, n_cells, sphere_radius, cell_radius):
    """Prepares faces to represent amountness in a spherical form."""
    paths = []
    for layer_idx in range(n_layers):
        for cell_idx in range(n_cells):
            x = np.cos(cell_idx / (n_cells / (np.pi * 2))) * (
                sphere_radius + layer_idx * cell_radius * 2
            )
            y = np.sin(cell_idx / (n_cells / (np.pi * 2))) * (
                sphere_radius + layer_idx * cell_radius * 2
            )
            circle = patches.Circle((x, y), radius=cell_radius)
            paths.append(circle)
    face_collection = clt.PatchCollection(
        paths, edgecolors="green", facecolors=["white"], linewidths=0.01
    )
    return face_collection


def update_face_collection(face_collection, data, n_layers, n_cells):
    """Updates faces from amountness data."""
    colors = []
    for layer_idx in range(n_layers):
        for cell_idx in range(n_cells):
            cell = data[layer_idx, cell_idx]
            colors.append(str(cell / (1 + cell)))
    face_collection.set_facecolors(colors)


def generate_arrow_collection(di, dj, n_layers, n_cells, sphere_radius, cell_radius):
    """Generates arrows situated spherically and pointing to the directions of given directions."""
    arrows = []
    for layer_idx in range(n_layers):
        for cell_idx in range(n_cells):
            angle = cell_idx / (n_cells / (np.pi * 2))
            x = np.cos(angle) * (sphere_radius + layer_idx * cell_radius * 2)
            y = np.sin(angle) * (sphere_radius + layer_idx * cell_radius * 2)

            # horizontal derivatives
            dx = dj[layer_idx, cell_idx] * 5

            # vertical derivatives
            dy = -(di[layer_idx, cell_idx] * 5)
            # dy = 0

            arrow = patches.Arrow(
                x,
                y,
                np.cos(angle) * dy + np.sin(angle) * dx,
                np.sin(angle) * dy - np.cos(angle) * dx,
                width=0.3,
            )
            arrows.append(arrow)

    arrow_collection = clt.PatchCollection(
        arrows,
        color="blue",
    )
    return arrow_collection


def finite_difference(state):
    """Computes finite difference dx, dy for every cell. Cells are assumed to form a circle
    and thus wrap around."""
    di = np.empty(state.shape)
    dj = np.empty(state.shape)
    for i in range(state.shape[0]):
        for j in range(state.shape[1]):
            # horizontal
            if j == 0:
                dj[i, j] = (state[i, 1] - state[i, -1]) / 2
            elif j == state.shape[1] - 1:
                dj[i, j] = (state[i, 0] - state[i, -2]) / 2
            else:
                dj[i, j] = (state[i, j + 1] - state[i, j - 1]) / 2

            if state.shape[0] == 0:
                raise Exception("One-layer case not handled yet.")

            # vertical
            if i == 0:
                di[i, j] = state[1, j] - state[0, j]
            elif i == state.shape[0] - 1:
                di[i, j] = state[-1, j] - state[-2, j]
            else:
                di[i, j] = (state[i + 1, j] - state[i - 1, j]) / 2
    return di, dj


def apply_diffusion(state, rule):
    """Calls cell automaton function for diffusion"""
    return rule.apply(State(state)).data


def simulate_sun(energy, n_cells):
    """Add randomly some energy to specific inner cells."""
    new_energy = energy.copy()
    added_total = 0

    # The sun
    added = np.max([0, random.random()*5 - 3])
    added_total += added
    new_energy[0, n_cells // 4] = energy[0, n_cells // 4] + added

    ## "Second sun"
    # added = np.max([0, random.random()*5 - 3])
    # added_total += added
    # new_energy[0, (n_cells * 3) // 4] = energy[0, (n_cells * 3) // 4] + added

    return new_energy, added_total


def simulate_escape(energy, amount):
    """Simulate energy escaping the atmosphere. High energy areas
    dissipate faster."""
    new_energy = energy.copy()
    difference = (energy[-1, :] * (amount / np.sum(energy[-1, :])))
    new_energy[-1, :] = new_energy[-1, :] - difference
    return new_energy


def simulate_upthrust(density, energy, n_layers, n_cells):
    """Move density vertically between cells based on the energy content.
    """
    new_density = density.copy()
    for i in range(n_layers - 1):
        for j in range(n_cells):

            sum_before = new_density[i, j] + new_density[i + 1, j]

            # from lower to higher, emphasizing lower layers.
            layer_factor = (n_layers - i) / n_layers
            change = (energy[i, j] - 1) * 0.2 * layer_factor + 1

            new_density[i, j] = new_density[i, j] / change
            new_density[i + 1, j] = new_density[i + 1, j] * change

            sum_after = new_density[i, j] + new_density[i + 1, j]

            # ensure that the change is conserving energy
            new_density[i, j] = new_density[i, j] * (sum_before / sum_after)
            new_density[i + 1, j] = new_density[i + 1, j] * (sum_before / sum_after)

    return new_density


if __name__ == "__main__":
    """Run as a script."""

    n_cells = 45
    n_layers = 7

    sphere_radius = 10
    cell_radius = 0.5

    # density = np.random.random((n_layers, n_cells)) * 2
    density = np.ones((n_layers, n_cells)) * 1.0
    energy = np.ones((n_layers, n_cells)) * 1.0

    # Create the diffusion rule instance
    diffusion_rule = RuleDiffusion(von_neumann=True, renormalize=True)

    # Create a tickless figure with two subplots
    fig, (ax_density, ax_energy) = plt.subplots(ncols=2)
    ax_density = init_plot(ax_density, sphere_radius, "Density")
    ax_energy = init_plot(ax_energy, sphere_radius, "Energy")

    # create initial configuration of density faces
    density_face_collection = init_face_collection(
        n_layers, n_cells, sphere_radius, cell_radius)
    ax_density.add_collection(density_face_collection)

    # create initial configuration of density wind arrows by random
    density_arrow_collection = generate_arrow_collection(
        0.1*np.random.random((n_layers, n_cells)),
        0.1*np.random.random((n_layers, n_cells)),
        n_layers, n_cells, sphere_radius, cell_radius)
    ax_density.add_collection(density_arrow_collection)

    # create initial configuration of energy faces
    energy_face_collection = init_face_collection(
        n_layers, n_cells, sphere_radius, cell_radius)
    ax_energy.add_collection(energy_face_collection)

    # "main loop"
    def animation_func(idx):
        """Helper to update a single frame."""

        # Get the global state
        global energy
        global density

        # Add energy to some inner cells to simulate sun (adding some extra energy to system)
        energy, added_total = simulate_sun(energy, n_cells)

        # Allow some energy to escape from outer cells (removing the same amount from the system)
        energy = simulate_escape(energy, added_total)

        # Simulate upthrust (keeps total density same)
        density = simulate_upthrust(density, energy, n_layers, n_cells)

        # Update energy by diffusion (keeps total energy same)
        energy = apply_diffusion(energy, diffusion_rule)

        # Update density by diffusion (keeps total density same)
        density = apply_diffusion(density, diffusion_rule)

        # compute derivatives between cells, representing winds
        di_density, dj_density = finite_difference(density)
        di_energy, dj_energy = finite_difference(energy)

        # natural choices for winds
        vertical_wind = -di_density
        horizontal_wind = dj_density

        # Update arrow visualization in the plot
        global density_arrow_collection
        density_arrow_collection.remove()
        density_arrow_collection = generate_arrow_collection(vertical_wind, horizontal_wind, n_layers, n_cells, sphere_radius, cell_radius)
        ax_density.add_collection(density_arrow_collection)

        # Update density colors
        update_face_collection(density_face_collection, density, n_layers, n_cells)

        # Update energy colors
        update_face_collection(energy_face_collection, energy, n_layers, n_cells)

    # Let matplotlib do the real work
    ani = FuncAnimation(fig, animation_func, frames=10000, interval=500, blit=False)
    plt.show()

