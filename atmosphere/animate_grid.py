# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib"]
# ///
"""Generate MP4 animation of grid atmosphere simulation.

Run:  uv run cellsimulations/atmosphere/animate_grid.py
Then: nix shell nixpkgs#ffmpeg --command ffmpeg -framerate 30 -i cellsimulations/atmosphere/frames_grid/f_%04d.png -c:v libx264 -pix_fmt yuv420p cellsimulations/atmosphere/grid.mp4
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from physics import GridGeometry, Params, step_grid, diagnostics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -- Config --
N_LAT, N_LON = 36, 72
geo = GridGeometry(N_LAT, N_LON)
p = Params()
OMEGA = 0.4

TOTAL_FRAMES = 600
SUBSTEPS = 5
SUN_SPEED = 0.01
FRAMES_DIR = os.path.join(os.path.dirname(__file__), 'frames_grid')

# -- State --
T = np.zeros(geo.shape)
u = np.zeros(geo.shape)
v = np.zeros(geo.shape)

os.makedirs(FRAMES_DIR, exist_ok=True)
sun = 0.0

lon_deg = np.degrees(geo.lon_grid)
lat_deg = np.degrees(geo.lat_grid)
skip = (slice(None, None, 2), slice(None, None, 3))

print(f"Grid {N_LAT}x{N_LON}, generating {TOTAL_FRAMES} frames...")
for frame in range(TOTAL_FRAMES):
    for _ in range(SUBSTEPS):
        T, u, v = step_grid(T, u, v, geo, p, sun, omega=OMEGA)
        sun += SUN_SPEED

    d = diagnostics(T, u, v)
    if frame % 100 == 0:
        print(f"  {frame:4d}/{TOTAL_FRAMES} | E={d['total']:8.1f} wind={d['max_wind']:.3f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    vmax = max(abs(T.min()), abs(T.max()), 0.01)
    ax1.imshow(T, cmap='RdBu_r', origin='lower', extent=[0,360,-90,90],
               aspect='auto', vmin=-vmax, vmax=vmax)
    ws = max(0.01, np.max(np.sqrt(u**2 + v**2)))
    ax1.quiver(lon_deg[skip], lat_deg[skip], u[skip], v[skip],
               scale=ws*40, color='k', alpha=0.6, width=0.003)
    ax1.set(xlabel='Lon', ylabel='Lat', title=f'Temperature + Wind')

    ax2.imshow(np.sqrt(u**2 + v**2), cmap='viridis', origin='lower',
               extent=[0,360,-90,90], aspect='auto')
    ax2.set(xlabel='Lon', ylabel='Lat', title=f'Wind Speed')

    fig.suptitle(f'Grid Atmosphere — frame {frame}  E={d["total"]:.0f}', fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(FRAMES_DIR, f'f_{frame:04d}.png'), dpi=80)
    plt.close(fig)

print(f"Done. {TOTAL_FRAMES} frames in {FRAMES_DIR}/")
print(f"To make MP4:")
print(f"  nix shell nixpkgs#ffmpeg --command ffmpeg -framerate 30 -i {FRAMES_DIR}/f_%04d.png -c:v libx264 -pix_fmt yuv420p cellsimulations/atmosphere/grid.mp4")
