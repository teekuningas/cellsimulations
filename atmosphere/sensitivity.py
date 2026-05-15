# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib"]
# ///
"""Sensitivity analysis toolkit for atmosphere simulations.

Provides reusable building blocks for systematic parameter studies:

    sweep_param()         — run simulation across a range of one parameter
    run_variants()        — run multiple step-function variants side-by-side
    plot_energy_curves()  — convergence curves, one line per run
    plot_T_heatmaps()     — final-state temperature fields, side-by-side
    plot_summary()        — asymptotic energy + max-wind bar charts

All plots land in output_analysis/ by default.

Convergence is measured over the last full solar orbit rather than a
fixed window — the sun completes one circuit every ~126 frames at the
default (sun_speed=0.01, substeps=5), so last-126-frame statistics give
a phase-averaged equilibrium estimate.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import replace

# One solar orbit in frames: 2π / (sun_speed * substeps) ≈ 2π / 0.05 ≈ 126
SOLAR_PERIOD_FRAMES = 126


# ═══════════════════════════════════════════════════════════
#  Simulation runners
# ═══════════════════════════════════════════════════════════

def sweep_param(step_fn, geo, base_params, param_name, values,
                n_steps=500, substeps=5, sun_speed=0.01,
                verbose=True, **step_kwargs):
    """Run simulation to convergence for each value of one parameter.

    Returns a list of result dicts, one per value:
        param_name    — name of the swept parameter
        param_value   — value used for this run
        history       — list of diagnostics dicts (one per frame)
        asymptotic    — last-solar-cycle mean of each diagnostic
        final_T/q1/q2 — field arrays at the last frame
    """
    from physics import diagnostics
    results = []
    for val in values:
        p = replace(base_params, **{param_name: val})
        T  = np.zeros(geo.shape)
        q1 = np.zeros(geo.shape)
        q2 = np.zeros(geo.shape)
        sun = 0.0
        history = []

        for _ in range(n_steps):
            for _s in range(substeps):
                T, q1, q2 = step_fn(T, q1, q2, geo, p, sun, **step_kwargs)
                sun += sun_speed
            history.append(diagnostics(T, q1, q2))

        asym = _asymptotic(history)
        if verbose:
            print(f"  {param_name}={val:.4g}: "
                  f"E∞={asym['total']:.1f}  wind∞={asym['max_wind']:.3f}")

        results.append({
            'param_name':  param_name,
            'param_value': val,
            'history':     history,
            'asymptotic':  asym,
            'final_T':     T.copy(),
            'final_q1':    q1.copy(),
            'final_q2':    q2.copy(),
        })
    return results


def run_variants(variants, geo, params,
                 n_steps=500, substeps=5, sun_speed=0.01,
                 static_sun=False, verbose=True,
                 diag_fn=None):
    """Run multiple step-function variants and collect results.

    variants   — list of (step_fn, label, extra_kwargs_dict) tuples
    static_sun — if True sun angle stays fixed at 0 (easier to isolate mechanism)
    diag_fn    — optional replacement for physics.diagnostics (e.g. ring_diagnostics)

    Returns list of result dicts: label, history, asymptotic, final_T/q1/q2.
    """
    from physics import diagnostics as default_diag
    measure = diag_fn if diag_fn is not None else default_diag

    results = []
    for step_fn, label, extra_kwargs in variants:
        T  = np.zeros(geo.shape)
        q1 = np.zeros(geo.shape)
        q2 = np.zeros(geo.shape)
        sun = 0.0
        history = []

        for _ in range(n_steps):
            for _s in range(substeps):
                T, q1, q2 = step_fn(T, q1, q2, geo, params, sun, **extra_kwargs)
                if not static_sun:
                    sun += sun_speed
            history.append(measure(T, q1, q2))

        asym = _asymptotic(history, static=static_sun)
        if verbose:
            print(f"  {label}: E∞={asym['total']:.1f}  wind∞={asym['max_wind']:.3f}")

        results.append({
            'label':      label,
            'history':    history,
            'asymptotic': asym,
            'final_T':    T.copy(),
            'final_q1':   q1.copy(),
            'final_q2':   q2.copy(),
        })
    return results


def _asymptotic(history, static=False):
    """Mean of last solar cycle (or 50 frames if static sun) as equilibrium estimate."""
    n = SOLAR_PERIOD_FRAMES if not static else 50
    tail = history[-min(n, len(history)):]
    return {k: float(np.mean([d[k] for d in tail])) for k in tail[0]}


# ═══════════════════════════════════════════════════════════
#  Plots
# ═══════════════════════════════════════════════════════════

def _label_of(r):
    """Extract display label from a result dict (sweep value or variant label)."""
    v = r.get('param_value')
    if v is not None:
        return f'{v:.3g}'
    return str(r.get('label', '?'))


def plot_energy_curves(results, key_label, output_path, field='total', title=None):
    """One energy-convergence curve per result, colored by parameter value.

    field — which diagnostics key to plot on the y-axis ('total', 'thermal',
            'kinetic', 'vert_ke', 'heat_transport', …)
    """
    n = len(results)
    cmap = plt.cm.viridis
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, r in enumerate(results):
        color = cmap(i / max(n - 1, 1))
        hist  = [d[field] for d in r['history']]
        ax.plot(hist, color=color, label=_label_of(r), alpha=0.85, lw=1.5)

    ax.set(xlabel='Frame', ylabel=field,
           title=title or f'{field} convergence — {key_label}')
    ax.legend(title=key_label, fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)


def plot_T_heatmaps(results, key_label, output_path, title=None, show_wind=True):
    """Side-by-side final temperature heatmaps with optional wind arrows.

    Rows = y-axis of the field array (lat for grid, altitude for ring).
    Cols = x-axis (lon for grid, azimuth for ring).
    Wind arrows (q1=u, q2=v/w) are subsampled for readability.
    """
    n     = len(results)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows),
                             squeeze=False)

    for i, r in enumerate(results):
        ax = axes[i // ncols][i % ncols]
        T  = r['final_T']
        vmax = max(abs(T.min()), abs(T.max()), 0.01)
        ax.imshow(T, cmap='RdBu_r', origin='lower', aspect='auto',
                  vmin=-vmax, vmax=vmax)

        if show_wind:
            q1, q2 = r['final_q1'], r['final_q2']
            # subsample so arrows don't overlap
            sy = max(1, T.shape[0] // 8)
            sx = max(1, T.shape[1] // 12)
            rows_s = np.arange(0, T.shape[0], sy)
            cols_s = np.arange(0, T.shape[1], sx)
            C, R   = np.meshgrid(cols_s, rows_s)
            ws = max(0.01, np.sqrt(np.max(q1 ** 2 + q2 ** 2)))
            ax.quiver(C, R, q1[::sy, ::sx], q2[::sy, ::sx],
                      scale=ws * 30, color='k', alpha=0.55, width=0.004)

        asym = r['asymptotic']['total']
        ax.set_title(f'{key_label}={_label_of(r)}\nE∞={asym:.0f}', fontsize=9)
        ax.axis('off')

    for i in range(n, nrows * ncols):
        axes[i // ncols][i % ncols].set_visible(False)

    fig.suptitle(title or f'Final temperature — {key_label}', fontsize=12)
    plt.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)


def plot_summary(results, key_label, output_path, title=None):
    """Bar charts of asymptotic total energy and max wind speed per run."""
    labels   = [_label_of(r) for r in results]
    energies = [r['asymptotic']['total']    for r in results]
    winds    = [r['asymptotic']['max_wind'] for r in results]
    x = np.arange(len(results))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.bar(x, energies, color='steelblue', alpha=0.8)
    ax1.set(xticks=x, xticklabels=labels, xlabel=key_label,
            ylabel='Asymptotic total energy', title='Equilibrium energy')
    ax1.grid(True, alpha=0.3, axis='y')

    ax2.bar(x, winds, color='coral', alpha=0.8)
    ax2.set(xticks=x, xticklabels=labels, xlabel=key_label,
            ylabel='Asymptotic max wind', title='Equilibrium wind speed')
    ax2.grid(True, alpha=0.3, axis='y')

    fig.suptitle(title or f'Equilibrium statistics — {key_label}', fontsize=12)
    plt.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)
