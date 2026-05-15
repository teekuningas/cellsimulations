# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib"]
# ///
"""Combined 3D model — Grid (lat×lon) + Vertical layers + Coriolis + Gravity.

Each altitude layer has horizontal circulation (pressure + Coriolis),
connected vertically by pressure + gravity. The same 7-step rules at
every cell, now in three dimensions.

A single ROTATION parameter drives both Coriolis and the day/night cycle —
they are the same physical phenomenon (planet rotation). Axial tilt sets
the sun's declination (0 = equinox, 23.5° = solstice).

Run:  uv run cellsimulations/atmosphere/run_sphere.py
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from physics import Params

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════

N_ALT = 8           # altitude layers (0=ground, 7=ceiling/tropopause)
N_LAT = 24          # latitude cells (pole to pole)
N_LON = 48          # longitude cells (periodic)

# Earth's rotation — ONE parameter drives both Coriolis and day/night cycle.
# This is the angular velocity of the planet. Higher = stronger Coriolis
# AND faster day/night cycle (they are the same physical phenomenon).
ROTATION = 0.25     # rad/time-unit (Coriolis omega AND sun angular speed)

G = 0.06            # gravity

# Axial tilt — sun's declination angle.
# 23.5° = northern hemisphere summer solstice (sun directly over Tropic of Cancer)
TILT = np.radians(0.0)    # 0 = equinox (symmetric), 23.5 = NH summer solstice

TOTAL_FRAMES = 600
SUBSTEPS = 6
SNAPSHOT_EVERY = 30

OUTPUT = os.path.join(os.path.dirname(__file__), 'output_sphere')
p = Params(solar=0.18, cooling=0.02, c_sq=0.15, diffuse=0.0, drag=0.02)


# ═══════════════════════════════════════════════════════════
#  3D Geometry
# ═══════════════════════════════════════════════════════════

class SphereGeometry:
    """Lat × lon × altitude. Combines grid and ring geometry."""

    def __init__(self, n_alt, n_lat, n_lon):
        self.n_alt = n_alt
        self.n_lat = n_lat
        self.n_lon = n_lon
        self.shape = (n_alt, n_lat, n_lon)

        lat = np.linspace(-np.pi / 2, np.pi / 2, n_lat)
        lon = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)

        # 2D lat-lon grids (for each altitude layer)
        self.lat_2d, self.lon_2d = np.meshgrid(lat, lon, indexing='ij')
        self.sin_lat = np.sin(self.lat_2d)   # (n_lat, n_lon)
        self.cos_lat = np.cos(self.lat_2d)   # (n_lat, n_lon)

    def solar(self, sun_lon, tilt=0.0):
        """Sun heats ground layer only. With axial tilt, sun is offset from equator.

        Full formula: Q ∝ max(0, sin(lat)*sin(dec) + cos(lat)*cos(dec)*cos(lon-sun))
        where dec = tilt (declination of the sun above/below equator).
        """
        pattern = np.zeros(self.shape)
        cos_zenith = (np.sin(self.lat_2d) * np.sin(tilt) +
                      np.cos(self.lat_2d) * np.cos(tilt) *
                      np.cos(self.lon_2d - sun_lon))
        pattern[0] = np.maximum(0.0, cos_zenith)
        return pattern


# ═══════════════════════════════════════════════════════════
#  3D Physics Step
# ═══════════════════════════════════════════════════════════

def _horiz_gradient(f_2d):
    """Central difference on 2D (lat, lon). Periodic in lon, bounded in lat."""
    # d/dlon (periodic)
    dfdx = (np.roll(f_2d, -1, axis=1) - np.roll(f_2d, 1, axis=1)) / 2
    # d/dlat (bounded)
    dfdy = np.zeros_like(f_2d)
    dfdy[1:-1] = (f_2d[2:] - f_2d[:-2]) / 2
    dfdy[0] = f_2d[1] - f_2d[0]
    dfdy[-1] = f_2d[-1] - f_2d[-2]
    return dfdx, dfdy


def _vert_gradient(f_3d):
    """Vertical gradient dT/dz. Bounded top and bottom."""
    dfdz = np.zeros_like(f_3d)
    dfdz[1:-1] = (f_3d[2:] - f_3d[:-2]) / 2
    dfdz[0] = f_3d[1] - f_3d[0]
    dfdz[-1] = f_3d[-1] - f_3d[-2]
    return dfdz


def _advect_2d(field, u, v, dt):
    """Upwind advection on a 2D horizontal slice."""
    # x-direction (periodic in lon)
    dfdx = np.where(
        u > 0,
        field - np.roll(field, 1, axis=1),
        np.roll(field, -1, axis=1) - field,
    )
    # y-direction (bounded in lat)
    dfdy_back = np.zeros_like(field)
    dfdy_back[1:] = field[1:] - field[:-1]
    dfdy_fwd = np.zeros_like(field)
    dfdy_fwd[:-1] = field[1:] - field[:-1]
    dfdy = np.where(v > 0, dfdy_back, dfdy_fwd)

    return field - dt * (u * dfdx + v * dfdy)


def _advect_vertical(f_3d, w, dt):
    """Upwind advection in the vertical direction only."""
    # w > 0 means upward
    dfdz_back = np.zeros_like(f_3d)
    dfdz_back[1:] = f_3d[1:] - f_3d[:-1]
    dfdz_fwd = np.zeros_like(f_3d)
    dfdz_fwd[:-1] = f_3d[1:] - f_3d[:-1]
    dfdz = np.where(w > 0, dfdz_back, dfdz_fwd)

    return f_3d - dt * w * dfdz


def step_sphere(T, u, v, w, geo, p, sun_lon, omega=0.4, g=0.06, tilt=0.0):
    """One timestep of the full 3D model.

    Pipeline: Heat → Cool → Pressure(H+V) → Coriolis → Gravity → Advect → Friction
    Same 7 rules as the 2D models, extended to 3 dimensions.

    omega — planet rotation rate (drives Coriolis deflection)
    tilt  — axial tilt / solar declination (0 = equinox, 23.5° = solstice)
    """
    # 1. Heat — sun warms the ground layer
    T = T + p.dt * p.solar * geo.solar(sun_lon, tilt=tilt)

    # 2. Cool — radiation to space (all layers)
    T = T * (1 - p.dt * p.cooling)

    # 3a. Horizontal pressure — ∇T → u, v (at each altitude)
    for z in range(geo.n_alt):
        dTdx, dTdy = _horiz_gradient(T[z])
        u[z] = u[z] - p.dt * p.c_sq * dTdx
        v[z] = v[z] - p.dt * p.c_sq * dTdy

    # 3b. Vertical pressure — dT/dz → w
    dTdz = _vert_gradient(T)
    w = w - p.dt * p.c_sq * dTdz

    # 4. Coriolis — rotation deflects horizontal wind (at each altitude)
    f = 2 * omega * geo.sin_lat * p.dt  # (n_lat, n_lon)
    c, s = np.cos(f), np.sin(f)
    for z in range(geo.n_alt):
        u_old, v_old = u[z].copy(), v[z].copy()
        u[z] = u_old * c + v_old * s
        v[z] = -u_old * s + v_old * c

    # 5. Gravity — constant downward pull on vertical wind
    w = w - p.dt * g

    # 6. Advection — wind carries everything
    # Horizontal advection (layer by layer)
    u0, v0, w0 = u.copy(), v.copy(), w.copy()
    for z in range(geo.n_alt):
        T[z] = _advect_2d(T[z], u0[z], v0[z], p.dt)
        u[z] = _advect_2d(u[z], u0[z], v0[z], p.dt)
        v[z] = _advect_2d(v[z], u0[z], v0[z], p.dt)
        w[z] = _advect_2d(w[z], u0[z], v0[z], p.dt)

    # Vertical advection
    T = _advect_vertical(T, w0, p.dt)
    u = _advect_vertical(u, w0, p.dt)
    v = _advect_vertical(v, w0, p.dt)
    w = _advect_vertical(w, w0, p.dt)

    # 7. Friction
    u = u * (1 - p.dt * p.drag)
    v = v * (1 - p.dt * p.drag)
    w = w * (1 - p.dt * p.drag)

    # Boundaries
    w[0, :, :] = 0     # no flow through ground
    w[-1, :, :] = 0    # no flow through tropopause
    v[:, 0, :] = 0     # no flow through south pole
    v[:, -1, :] = 0    # no flow through north pole

    return T, u, v, w


# ═══════════════════════════════════════════════════════════
#  Visualization
# ═══════════════════════════════════════════════════════════

def render_snapshot(T, u, v, w, geo, frame, sun_lon):
    """Multi-panel figure showing the 3D circulation."""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    lats = np.linspace(-90, 90, geo.n_lat)
    lons_1d = np.linspace(0, 360, geo.n_lon, endpoint=False)
    lons_2d, lats_2d = np.meshgrid(lons_1d, lats)

    # Skip pattern for quiver
    skip = (slice(None, None, 2), slice(None, None, 3))

    # --- Row 1: Wind maps at 3 altitudes ---
    alt_levels = [0, geo.n_alt // 2, geo.n_alt - 2]
    alt_names = ['Surface (z=0)', f'Mid-level (z={alt_levels[1]})',
                 f'Upper (z={alt_levels[2]})']

    for i, (z, name) in enumerate(zip(alt_levels, alt_names)):
        ax = fig.add_subplot(gs[0, i])
        speed = np.sqrt(u[z]**2 + v[z]**2)
        vmax_s = max(speed.max(), 0.01)
        im = ax.pcolormesh(lons_1d, lats, speed, cmap='viridis',
                          shading='auto', vmin=0, vmax=vmax_s)
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

        ws = max(0.01, np.sqrt(u[z]**2 + v[z]**2).max())
        ax.quiver(lons_2d[skip], lats_2d[skip], u[z][skip], v[z][skip],
                  scale=ws * 30, color='white', alpha=0.7, width=0.004,
                  headwidth=4, headlength=3)
        ax.set(title=name, xlim=(0, 360), ylim=(-90, 90))
        if i == 0:
            ax.set_ylabel('Latitude (°)')

    # --- Row 2: The money plots ---
    # Panel: Zonal-mean u-wind vs altitude and latitude
    ax_main = fig.add_subplot(gs[1, :2])
    u_zonal = u.mean(axis=2)  # average over longitude → (n_alt, n_lat)
    vmax_u = max(np.abs(u_zonal).max(), 0.01)
    im = ax_main.pcolormesh(lats, np.arange(geo.n_alt), u_zonal,
                            cmap='RdBu_r', shading='auto',
                            vmin=-vmax_u, vmax=vmax_u)
    plt.colorbar(im, ax=ax_main, label='Zonal wind (u)')
    ax_main.set(xlabel='Latitude (°)', ylabel='Altitude layer',
                title='Zonal-mean u(lat, alt)')
    ax_main.axhline(0, color='k', lw=0.3)
    ax_main.axvline(0, color='k', lw=0.3, ls='--')

    # Panel: Wind profiles at specific latitudes
    ax_prof = fig.add_subplot(gs[1, 2])
    lat_indices = [geo.n_lat // 6, geo.n_lat // 3, geo.n_lat // 2,
                   2 * geo.n_lat // 3, 5 * geo.n_lat // 6]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for idx, color in zip(lat_indices, colors):
        lat_val = lats[idx]
        u_profile = u_zonal[:, idx]
        ax_prof.plot(u_profile, np.arange(geo.n_alt), color=color, lw=1.5,
                    label=f'{lat_val:.0f}°')
    ax_prof.axvline(0, color='k', lw=0.5, ls='--')
    ax_prof.set(xlabel='Zonal wind (u)', ylabel='Altitude',
                title='u(z) at selected latitudes')
    ax_prof.legend(fontsize=8, loc='best')
    ax_prof.grid(True, alpha=0.2)

    # --- Row 3: Cross-sections ---
    # Meridional cross-section (lon-averaged): v, w vs lat, alt
    ax_cross = fig.add_subplot(gs[2, 0])
    v_zonal = v.mean(axis=2)
    w_zonal = w.mean(axis=2)
    speed_cross = np.sqrt(v_zonal**2 + w_zonal**2)
    ax_cross.pcolormesh(lats, np.arange(geo.n_alt), speed_cross,
                       cmap='Oranges', shading='auto')
    # Streamplot of (v, w) in the meridional plane
    lat_fine = np.linspace(-90, 90, geo.n_lat)
    alt_fine = np.arange(geo.n_alt)
    if speed_cross.max() > 1e-6:
        ax_cross.quiver(lats[::2], alt_fine, v_zonal[:, ::2], w_zonal[:, ::2],
                       scale=speed_cross.max() * 15, color='k', alpha=0.6,
                       width=0.005)
    ax_cross.set(xlabel='Latitude (°)', ylabel='Altitude',
                title='Meridional circulation (v, w)')

    # Temperature cross-section
    ax_T = fig.add_subplot(gs[2, 1])
    T_zonal = T.mean(axis=2)
    vmax_T = max(np.abs(T_zonal).max(), 0.01)
    im = ax_T.pcolormesh(lats, np.arange(geo.n_alt), T_zonal,
                         cmap='hot', shading='auto')
    plt.colorbar(im, ax=ax_T, fraction=0.04, label='T')
    ax_T.set(xlabel='Latitude (°)', ylabel='Altitude',
             title='Temperature T(lat, alt)')

    # Vertical wind cross-section
    ax_w = fig.add_subplot(gs[2, 2])
    vmax_w = max(np.abs(w_zonal).max(), 0.01)
    im = ax_w.pcolormesh(lats, np.arange(geo.n_alt), w_zonal,
                         cmap='RdBu_r', shading='auto',
                         vmin=-vmax_w, vmax=vmax_w)
    plt.colorbar(im, ax=ax_w, fraction=0.04, label='w')
    ax_w.set(xlabel='Latitude (°)', ylabel='Altitude',
             title='Vertical wind w(lat, alt)')

    fig.suptitle(f'3D Atmosphere — frame {frame}  |  '
                 f'{geo.n_alt} alt × {geo.n_lat} lat × {geo.n_lon} lon  |  '
                 f'rotation={ROTATION}  g={G}  tilt={np.degrees(TILT):.1f}°',
                 fontsize=13, fontweight='bold')
    return fig


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def main():
    geo = SphereGeometry(N_ALT, N_LAT, N_LON)
    T = np.zeros(geo.shape)
    u = np.zeros(geo.shape)
    v = np.zeros(geo.shape)
    w = np.zeros(geo.shape)

    os.makedirs(OUTPUT, exist_ok=True)
    sun = 0.0
    energy_hist = []

    # Sun advances by ROTATION * dt per physics step (same rotation drives Coriolis)
    sun_step = ROTATION * p.dt

    print(f"Sphere {N_ALT} alt × {N_LAT} lat × {N_LON} lon")
    print(f"  rotation={ROTATION}, g={G}, tilt={np.degrees(TILT):.1f}°")
    print(f"  sun_step={sun_step:.4f} rad/step (= rotation × dt)")
    print(f"  {TOTAL_FRAMES} frames × {SUBSTEPS} substeps = {TOTAL_FRAMES*SUBSTEPS} steps")
    print(f"  Params: solar={p.solar} cool={p.cooling} c²={p.c_sq} drag={p.drag}")
    print()

    for frame in range(TOTAL_FRAMES):
        for _ in range(SUBSTEPS):
            T, u, v, w = step_sphere(T, u, v, w, geo, p, sun,
                                     omega=ROTATION, g=G, tilt=TILT)
            sun += sun_step

        total_E = 0.5 * (np.sum(T**2) + np.sum(u**2) + np.sum(v**2) + np.sum(w**2))
        max_wind = np.max(np.sqrt(u**2 + v**2 + w**2))
        energy_hist.append(total_E)

        if frame % 50 == 0:
            u_zonal = u.mean(axis=2)
            u_surface = u_zonal[0, N_LAT // 2]
            u_upper = u_zonal[-2, N_LAT // 2]
            print(f"  frame {frame:4d} | E={total_E:8.1f} "
                  f"max_w={max_wind:.3f} "
                  f"u_surf@eq={u_surface:.4f} u_upper@eq={u_upper:.4f}")

        if frame % SNAPSHOT_EVERY == 0 or frame == TOTAL_FRAMES - 1:
            fig = render_snapshot(T, u, v, w, geo, frame, sun)
            fig.savefig(os.path.join(OUTPUT, f'sphere_{frame:04d}.png'), dpi=110)
            plt.close(fig)

    # Energy evolution
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(energy_hist, color='steelblue', lw=1.5)
    ax.set(xlabel='Frame', ylabel='Energy', title='Total energy')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT, 'energy.png'), dpi=100)
    plt.close(fig)

    # Final summary plot: zonal wind at different heights
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    u_zonal = u.mean(axis=2)
    lats = np.linspace(-90, 90, N_LAT)

    ax = axes[0]
    for z in range(N_ALT):
        alpha = 0.3 + 0.7 * z / (N_ALT - 1)
        ax.plot(lats, u_zonal[z], lw=1.5, alpha=alpha,
                label=f'z={z}' if z % 2 == 0 else None)
    ax.axhline(0, color='k', lw=0.5)
    ax.set(xlabel='Latitude (°)', ylabel='Zonal wind (u)',
           title='Final zonal wind by altitude')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    # Show u at equator and 45N as function of height
    eq_idx = N_LAT // 2
    mid_idx = 3 * N_LAT // 4
    ax.plot(u_zonal[:, eq_idx], range(N_ALT), 'b-o', lw=2, label='Equator')
    ax.plot(u_zonal[:, mid_idx], range(N_ALT), 'r-o', lw=2, label='45°N')
    ax.plot(u_zonal[:, N_LAT // 4], range(N_ALT), 'g-o', lw=2, label='45°S')
    ax.axvline(0, color='k', lw=0.5, ls='--')
    ax.set(xlabel='Zonal wind (u)', ylabel='Altitude',
           title='Wind profile vs height')
    ax.legend()
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT, 'summary_profiles.png'), dpi=110)
    plt.close(fig)

    print(f"\nDone. Output in {OUTPUT}/")
    print(f"  Final energy: {energy_hist[-1]:.1f}")
    print(f"\n  === WIND AT DIFFERENT HEIGHTS ===")
    print(f"  Zonal wind (lon-averaged) at equator:")
    for z in range(N_ALT):
        print(f"    z={z}: u = {u_zonal[z, eq_idx]:+.4f}")
    print(f"\n  Does wind reverse with height? ", end='')
    signs = np.sign(u_zonal[:, eq_idx])
    if np.any(signs[:-1] != signs[1:]):
        print("YES ✓")
    else:
        print("No (same direction at all heights)")


if __name__ == '__main__':
    main()
