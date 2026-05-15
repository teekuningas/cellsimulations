# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib"]
# ///
"""Gravity model validation — comparing vertical forcing mechanisms.

Four variants are compared:

  A  Gravity model (canonical) — symmetric pressure + gravity
       Pressure drives both u and w symmetrically.
       Gravity pulls w down constantly. Buoyancy-like circulation
       emerges from the competition without an explicit buoyancy rule.

  B  Pressure-only — symmetric pressure, no gravity
       Same as A but g=0. Demonstrates the net-upward bias that
       occurs when nothing counteracts the vertical pressure gradient.

  C  Legacy buoyancy — horizontal anomaly rule
       Original model: w += buoyancy * (T - layer_mean(T)).
       Requires diffusion to bootstrap heat into upper layers.

  D  Control — horizontal pressure only, no vertical forcing
       Only horizontal pressure drives u. Vertical wind evolves
       through advection alone.

Run: uv run analyse_gravity.py
Output: output_analysis/gravity_*.png
"""

import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from physics import (RingGeometry, Params,
                     step_ring, step_ring_legacy, step_ring_pressure_only,
                     ring_diagnostics)
from sensitivity import run_variants, plot_energy_curves, plot_T_heatmaps

OUTPUT = os.path.join(os.path.dirname(__file__), 'output_analysis')
os.makedirs(OUTPUT, exist_ok=True)

geo    = RingGeometry(8, 48)
params = Params(solar=0.25, cooling=0.03, diffuse=0.0)
params_with_diffusion = Params(solar=0.25, cooling=0.03, diffuse=0.1)

N_STEPS  = 500
SUBSTEPS = 5

# Variant definitions: (step_fn, label, extra_kwargs)
VARIANTS = [
    (step_ring,              'A: gravity (g=0.08)',       {'g': 0.08}),
    (step_ring_pressure_only,'B: pressure only (g=0)',    {}),
    (step_ring_legacy,       'C: legacy buoyancy',        {'buoyancy': 0.2}),
    (step_ring_legacy,       'D: control (no vert force)',{'buoyancy': 0.0}),
]

# Legacy buoyancy needs diffusion, so we run two batches:
# Batch 1: without diffusion (diffuse=0) — shows gravity model advantage
# Batch 2: with diffusion (diffuse=0.1) — level playing field

for label, p in [("no diffusion", params), ("with diffusion", params_with_diffusion)]:
    tag = label.replace(' ', '_')
    print(f"\n=== Gravity variants — static sun, {label} ===")
    res_static = run_variants(
        VARIANTS, geo, p,
        n_steps=N_STEPS, substeps=SUBSTEPS,
        static_sun=True,
        diag_fn=ring_diagnostics,
    )

    plot_energy_curves(res_static, 'variant',
        os.path.join(OUTPUT, f'gravity_static_{tag}_energy.png'),
        title=f'Gravity variants — static sun, {label}: total energy')

    plot_T_heatmaps(res_static, 'variant',
        os.path.join(OUTPUT, f'gravity_static_{tag}_states.png'),
        title=f'Gravity variants — static sun, {label}: final T + wind')

    # ── Vertical circulation diagnostics ──
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    colors = ['steelblue', 'darkorange', 'seagreen', 'crimson']

    for col, (r, c) in enumerate(zip(res_static, colors)):
        hist = r['history']

        ax = axes[0][col]
        vert   = [d['vert_ke']        for d in hist]
        htrans = [d['heat_transport'] for d in hist]
        ax.plot(vert,   color=c,          label='vertical KE',   lw=1.5)
        ax.plot(htrans, color=c, ls='--', label='heat transport', lw=1.5)
        ax.axhline(0, color='k', lw=0.5, alpha=0.4)
        ax.set_title(r['label'], fontsize=8)
        ax.set_xlabel('Frame', fontsize=7)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

        ax = axes[1][col]
        net_w = [d['net_w'] for d in hist]
        ax.plot(net_w, color=c, lw=1.5)
        ax.axhline(0, color='k', lw=0.5, alpha=0.4)
        ax.set_title(f'net vertical velocity', fontsize=8)
        ax.set_xlabel('Frame', fontsize=7)
        ax.grid(True, alpha=0.3)

    axes[0][0].set_ylabel('Energy / transport', fontsize=8)
    axes[1][0].set_ylabel('mean(w)', fontsize=8)
    fig.suptitle(f'Vertical circulation — static sun, {label}\n'
                 f'net_w ≈ 0 → balanced cells', fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT, f'gravity_static_{tag}_circulation.png'), dpi=100)
    plt.close(fig)

    # ── Summary table ──
    print(f"\n--- Asymptotic summary ({label}) ---")
    print(f"{'Variant':<30}  {'E∞':>8}  {'vert_KE∞':>9}  {'net_w∞':>8}  {'heat_tr∞':>9}  {'lapse':>7}")
    for r in res_static:
        a = r['asymptotic']
        lr = a.get('lapse_rate', float('nan'))
        print(f"{r['label']:<30}  {a['total']:>8.1f}  {a['vert_ke']:>9.2f}"
              f"  {a['net_w']:>8.4f}  {a['heat_transport']:>9.4f}  {lr:>7.3f}")


# ── Moving sun comparison (canonical gravity model) ──
print("\n=== Gravity model — moving sun (long run) ===")
res_moving = run_variants(
    [(step_ring, 'gravity (g=0.08)', {'g': 0.08})],
    geo, params,
    n_steps=1000, substeps=SUBSTEPS,
    static_sun=False,
    diag_fn=ring_diagnostics,
)
plot_energy_curves(res_moving, 'variant',
    os.path.join(OUTPUT, 'gravity_moving_energy.png'),
    title='Gravity model — moving sun (1000 frames): energy convergence')

print(f"\nDone. Plots → {OUTPUT}/gravity_*.png")
