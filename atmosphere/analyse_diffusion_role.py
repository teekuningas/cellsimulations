# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib"]
# ///
"""Diffusion role investigation — what does diffusion actually do?

The gravity model works without diffusion. So what role does diffusion
play when it IS present? This script decomposes diffusion into its
components to find out:

  1. No diffusion       — baseline (canonical gravity model)
  2. Full diffusion     — T + u + w all smoothed (traditional)
  3. Wind-only diffusion — only u and w smoothed (turbulent viscosity)
  4. T-only diffusion   — only T smoothed (thermal mixing)

For each, we measure:
  - Total energy (system intensity)
  - Vertical KE (circulation strength)
  - Heat transport (warm-air-rising efficiency)
  - Net vertical velocity (bias check)
  - Lapse rate (vertical temperature gradient)

We test at several diffusion strengths to find the collapse boundary.

Run: uv run analyse_diffusion_role.py
Output: output_analysis/diffusion_role_*.png
"""

import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from physics import RingGeometry, Params, step_ring, ring_diagnostics
from sensitivity import _asymptotic, SOLAR_PERIOD_FRAMES

OUTPUT = os.path.join(os.path.dirname(__file__), 'output_analysis')
os.makedirs(OUTPUT, exist_ok=True)

geo = RingGeometry(8, 48)
G = 0.08
N_STEPS  = 600
SUBSTEPS = 5

DIFFUSE_LEVELS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4]


def run_ring_diffusion_variant(diffuse_T, diffuse_wind, label,
                               n_steps=N_STEPS, moving_sun=True):
    """Run ring gravity model with specific diffusion decomposition."""
    p = Params(solar=0.25, cooling=0.03, diffuse=0.0)
    T  = np.zeros(geo.shape)
    u  = np.zeros(geo.shape)
    w  = np.zeros(geo.shape)
    sun = 0.0
    history = []

    for _ in range(n_steps):
        for _s in range(SUBSTEPS):
            T, u, w = step_ring(T, u, w, geo, p, sun, g=G,
                                diffuse_T=diffuse_T, diffuse_wind=diffuse_wind)
            if moving_sun:
                sun += 0.01
        history.append(ring_diagnostics(T, u, w))

    asym = _asymptotic(history)
    return {
        'label': label,
        'diffuse_T': diffuse_T,
        'diffuse_wind': diffuse_wind,
        'history': history,
        'asymptotic': asym,
        'final_T': T.copy(),
    }


# ═══════════════════════════════════════════════════════════
#  Experiment 1: Decomposed diffusion at fixed level
# ═══════════════════════════════════════════════════════════

print("=== Experiment 1: Diffusion decomposition (diffuse=0.05) ===")
D = 0.05
decomposed = [
    run_ring_diffusion_variant(0.0, 0.0, '1. none'),
    run_ring_diffusion_variant(D,   D,   '2. full (T+wind)'),
    run_ring_diffusion_variant(0.0, D,   '3. wind only'),
    run_ring_diffusion_variant(D,   0.0, '4. T only'),
]

print(f"\n{'Variant':<22}  {'E∞':>8}  {'vert_KE∞':>9}  {'net_w∞':>8}  {'heat_tr∞':>9}  {'lapse':>7}")
for r in decomposed:
    a = r['asymptotic']
    lr = a.get('lapse_rate', float('nan'))
    print(f"{r['label']:<22}  {a['total']:>8.1f}  {a['vert_ke']:>9.2f}"
          f"  {a['net_w']:>8.4f}  {a['heat_transport']:>9.4f}  {lr:>7.3f}")

# Plot energy convergence
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
colors = ['steelblue', 'darkorange', 'seagreen', 'crimson']
metrics = ['total', 'vert_ke', 'heat_transport', 'lapse_rate']
titles  = ['Total energy', 'Vertical KE', 'Heat transport (w·T)', 'Lapse rate (dT/dz)']

for ax, metric, title in zip(axes.flat, metrics, titles):
    for r, c in zip(decomposed, colors):
        vals = [d.get(metric, 0) for d in r['history']]
        ax.plot(vals, color=c, label=r['label'], lw=1.5, alpha=0.85)
    ax.set(xlabel='Frame', ylabel=metric, title=title)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

fig.suptitle(f'Diffusion decomposition (level={D}) — gravity ring model', fontsize=12)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT, 'diffusion_role_decomposed.png'), dpi=100)
plt.close(fig)


# ═══════════════════════════════════════════════════════════
#  Experiment 2: Collapse boundary — which component causes it?
# ═══════════════════════════════════════════════════════════

print("\n=== Experiment 2: Collapse boundary scan ===")

results_full = []
results_wind = []
results_T    = []

for d in DIFFUSE_LEVELS:
    results_full.append(run_ring_diffusion_variant(d, d, f'full d={d:.2f}'))
    results_wind.append(run_ring_diffusion_variant(0, d, f'wind d={d:.2f}'))
    results_T.append(run_ring_diffusion_variant(d, 0,    f'T d={d:.2f}'))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, results, title in zip(axes,
    [results_full, results_wind, results_T],
    ['Full diffusion (T+wind)', 'Wind-only diffusion', 'T-only diffusion']):

    energies = [r['asymptotic']['total'] for r in results]
    vert_ke  = [r['asymptotic']['vert_ke'] for r in results]
    x = DIFFUSE_LEVELS

    ax.plot(x, energies, 'o-', color='steelblue', label='total energy', lw=2)
    ax2 = ax.twinx()
    ax2.plot(x, vert_ke, 's--', color='coral', label='vertical KE', lw=2)

    ax.set(xlabel='diffuse level', ylabel='Total energy', title=title)
    ax2.set_ylabel('Vertical KE', color='coral')
    ax.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

fig.suptitle('Collapse boundary — which diffusion component causes collapse?\n'
             'Energy drop + vert_KE → 0 indicates collapse', fontsize=11)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT, 'diffusion_role_collapse.png'), dpi=100)
plt.close(fig)

# Print the summary
print(f"\n{'Type':<12}  {'diffuse':>8}  {'E∞':>8}  {'vert_KE':>8}  {'lapse':>7}")
for label, results in [('Full', results_full), ('Wind', results_wind), ('T-only', results_T)]:
    for r in results:
        a = r['asymptotic']
        lr = a.get('lapse_rate', float('nan'))
        print(f"{label:<12}  {r['diffuse_T'] + r['diffuse_wind']:>8.2f}  {a['total']:>8.1f}  {a['vert_ke']:>8.2f}  {lr:>7.3f}")
    print()


# ═══════════════════════════════════════════════════════════
#  Experiment 3: Temperature profile comparison
# ═══════════════════════════════════════════════════════════

print("=== Experiment 3: Temperature profiles ===")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, results, title in zip(axes,
    [results_full, results_wind, results_T],
    ['Full diffusion', 'Wind-only diffusion', 'T-only diffusion']):

    for r in results:
        T_profile = r['final_T'].mean(axis=1)  # average over azimuth
        ax.plot(T_profile, range(len(T_profile)), 'o-',
                label=f'd={r["diffuse_T"] + r["diffuse_wind"]:.2f}', lw=1.5)
    ax.set(xlabel='Mean temperature', ylabel='Layer (0=ground)', title=title)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

fig.suptitle('Vertical temperature profiles — diffusion decomposition\n'
             'Steeper = stronger gradient = more buoyancy-like force', fontsize=11)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT, 'diffusion_role_profiles.png'), dpi=100)
plt.close(fig)

print(f"\nDone. Plots → {OUTPUT}/diffusion_role_*.png")
