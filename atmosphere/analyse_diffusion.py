# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib"]
# ///
"""Diffusion sensitivity analysis.

Question: How does turbulent mixing rate (diffuse) affect atmospheric
          circulation strength and equilibrium energy?

Grid: uses standard step_grid (diffusion is the only smoothing mechanism)
Ring: uses gravity model (diffusion is optional — physics works without it)

Run: uv run analyse_diffusion.py
Output: output_analysis/diffusion_*.png
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from physics import GridGeometry, RingGeometry, Params, step_grid, step_ring
from sensitivity import sweep_param, plot_energy_curves, plot_T_heatmaps, plot_summary

OUTPUT = os.path.join(os.path.dirname(__file__), 'output_analysis')
os.makedirs(OUTPUT, exist_ok=True)

DIFFUSE_VALUES = [0.0, 0.02, 0.05, 0.10, 0.20, 0.40]
N_STEPS   = 500
SUBSTEPS  = 5


# ── Grid sweep ─────────────────────────────────────────────
print("=== Diffusion sweep — Grid ===")
geo_grid  = GridGeometry(36, 72)
p_grid    = Params()
results_g = sweep_param(
    step_grid, geo_grid, p_grid,
    'diffuse', DIFFUSE_VALUES,
    n_steps=N_STEPS, substeps=SUBSTEPS,
    omega=0.4,
)

plot_energy_curves(results_g, 'diffuse',
    os.path.join(OUTPUT, 'diffusion_grid_energy.png'),
    title='Diffusion sensitivity — Grid: total energy convergence')

plot_T_heatmaps(results_g, 'diffuse',
    os.path.join(OUTPUT, 'diffusion_grid_states.png'),
    title='Diffusion sensitivity — Grid: final temperature + wind')

plot_summary(results_g, 'diffuse',
    os.path.join(OUTPUT, 'diffusion_grid_summary.png'),
    title='Diffusion sensitivity — Grid: equilibrium statistics')


# ── Ring sweep (gravity model) ──────────────────────────────
print("=== Diffusion sweep — Ring (gravity model) ===")
geo_ring  = RingGeometry(8, 48)
p_ring    = Params(solar=0.25, cooling=0.03)
results_r = sweep_param(
    step_ring, geo_ring, p_ring,
    'diffuse', DIFFUSE_VALUES,
    n_steps=N_STEPS, substeps=SUBSTEPS,
    g=0.08,
)

plot_energy_curves(results_r, 'diffuse',
    os.path.join(OUTPUT, 'diffusion_ring_energy.png'),
    title='Diffusion sensitivity — Ring (gravity): total energy convergence')

plot_T_heatmaps(results_r, 'diffuse',
    os.path.join(OUTPUT, 'diffusion_ring_states.png'),
    title='Diffusion sensitivity — Ring (gravity): final T + wind\n(rows=altitude, cols=azimuth)')

plot_summary(results_r, 'diffuse',
    os.path.join(OUTPUT, 'diffusion_ring_summary.png'),
    title='Diffusion sensitivity — Ring (gravity): equilibrium statistics')

print(f"\nDone. 6 plots → {OUTPUT}/diffusion_*.png")
