# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib"]
# ///
"""Grid experiment — lat×lon atmospheric circulation.

Run:  uv run cellsimulations/atmosphere/run_grid.py
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from physics import GridGeometry, Params, step_grid, diagnostics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Config ─────────────────────────────────────────
N_LAT, N_LON = 36, 72
geo = GridGeometry(N_LAT, N_LON)
p = Params()
OMEGA = 0.4            # Coriolis strength (geometry-specific, not in Params)

TOTAL_FRAMES = 500
SUBSTEPS = 5
SNAPSHOT_EVERY = 50
SUN_SPEED = 0.01
OUTPUT = os.path.join(os.path.dirname(__file__), 'output_grid')

# ── Initial state ──────────────────────────────────
T = np.zeros(geo.shape)
u = np.zeros(geo.shape)
v = np.zeros(geo.shape)

# ── Run ────────────────────────────────────────────
os.makedirs(OUTPUT, exist_ok=True)
sun = 0.0
energy_hist = []

lon_deg = np.degrees(geo.lon_grid)
lat_deg = np.degrees(geo.lat_grid)
skip = (slice(None, None, 2), slice(None, None, 3))

print(f"Grid {N_LAT}×{N_LON}, {TOTAL_FRAMES} frames, params: {p}")
for frame in range(TOTAL_FRAMES):
    for _ in range(SUBSTEPS):
        T, u, v = step_grid(T, u, v, geo, p, sun, omega=OMEGA)
        sun += SUN_SPEED

    d = diagnostics(T, u, v)
    energy_hist.append(d['total'])

    if frame % 50 == 0:
        print(f"  {frame:4d} | E={d['total']:8.1f} wind={d['max_wind']:.3f} T̄={d['mean_T']:.3f}")

    if frame % SNAPSHOT_EVERY == 0 or frame == TOTAL_FRAMES - 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        vmax = max(abs(T.min()), abs(T.max()), 0.01)
        ax1.imshow(T, cmap='RdBu_r', origin='lower', extent=[0,360,-90,90],
                   aspect='auto', vmin=-vmax, vmax=vmax)
        ws = max(0.01, np.max(np.sqrt(u**2+v**2)))
        ax1.quiver(lon_deg[skip], lat_deg[skip], u[skip], v[skip],
                   scale=ws*40, color='k', alpha=0.6, width=0.003)
        ax1.set(xlabel='Lon', ylabel='Lat', title=f'T + wind [{frame}]')

        ax2.imshow(np.sqrt(u**2+v**2), cmap='viridis', origin='lower',
                   extent=[0,360,-90,90], aspect='auto')
        ax2.set(xlabel='Lon', ylabel='Lat', title=f'Wind speed [{frame}]')
        plt.tight_layout()
        fig.savefig(os.path.join(OUTPUT, f'grid_{frame:04d}.png'), dpi=100)
        plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(energy_hist, color='steelblue')
ax.set(xlabel='Frame', ylabel='Energy', title='Total energy')
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT, 'energy.png'), dpi=100)
plt.close(fig)
print("Done.")
