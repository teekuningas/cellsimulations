# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib"]
# ///
"""Friction sensitivity analysis.

Question: How does surface drag rate affect equilibrium wind speed and
          the overall circulation pattern?

Grid: standard step_grid with Coriolis
Ring: gravity model (symmetric pressure + gravity)

Run: uv run analyse_friction.py
Output: output_analysis/friction_*.png
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from physics import GridGeometry, RingGeometry, Params, step_grid, step_ring
from sensitivity import sweep_param, plot_energy_curves, plot_T_heatmaps, plot_summary

OUTPUT = os.path.join(os.path.dirname(__file__), 'output_analysis')
os.makedirs(OUTPUT, exist_ok=True)

DRAG_VALUES = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10]
N_STEPS  = 500
SUBSTEPS = 5


# ── Grid sweep ─────────────────────────────────────────────
print("=== Friction sweep — Grid ===")
geo_grid  = GridGeometry(36, 72)
p_grid    = Params()
results_g = sweep_param(
    step_grid, geo_grid, p_grid,
    'drag', DRAG_VALUES,
    n_steps=N_STEPS, substeps=SUBSTEPS,
    omega=0.4,
)

plot_energy_curves(results_g, 'drag',
    os.path.join(OUTPUT, 'friction_grid_energy.png'),
    title='Friction sensitivity — Grid: total energy convergence')

plot_T_heatmaps(results_g, 'drag',
    os.path.join(OUTPUT, 'friction_grid_states.png'),
    title='Friction sensitivity — Grid: final temperature + wind')

plot_summary(results_g, 'drag',
    os.path.join(OUTPUT, 'friction_grid_summary.png'),
    title='Friction sensitivity — Grid: equilibrium statistics')


# ── Ring sweep (gravity model) ──────────────────────────────
print("=== Friction sweep — Ring (gravity model) ===")
geo_ring  = RingGeometry(8, 48)
p_ring    = Params(solar=0.25, cooling=0.03, diffuse=0.0)
results_r = sweep_param(
    step_ring, geo_ring, p_ring,
    'drag', DRAG_VALUES,
    n_steps=N_STEPS, substeps=SUBSTEPS,
    g=0.08,
)

plot_energy_curves(results_r, 'drag',
    os.path.join(OUTPUT, 'friction_ring_energy.png'),
    title='Friction sensitivity — Ring (gravity): total energy convergence')

plot_T_heatmaps(results_r, 'drag',
    os.path.join(OUTPUT, 'friction_ring_states.png'),
    title='Friction sensitivity — Ring (gravity): final T + wind\n(rows=altitude, cols=azimuth)')

plot_summary(results_r, 'drag',
    os.path.join(OUTPUT, 'friction_ring_summary.png'),
    title='Friction sensitivity — Ring (gravity): equilibrium statistics')

print(f"\nDone. 6 plots → {OUTPUT}/friction_*.png")
