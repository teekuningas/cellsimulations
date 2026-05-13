# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""Atmosphere Simulation — Core Physics
=====================================

Three fields on a 2D surface:

    T  — temperature    (drives pressure)
    u  — wind along x   (east / azimuthal)
    v  — wind along y   (north / vertical)

Five operations per timestep, each one physical law:

    1. Heating    — sun adds energy at the surface
    2. Cooling    — radiation removes energy (∝ temperature)
    3. Pressure   — temperature gradient accelerates wind
    4. Advection  — wind carries heat and momentum
    5. Diffusion  — mixing smooths everything + friction damps wind

Geometry supplies one extra force:
    Grid → Coriolis (rotation deflects wind)
    Ring → Buoyancy (hot air rises)

Stability: bounded input, proportional removal, advection conserves,
diffusion dissipates → energy always converges to a finite value.

All operations are pure numpy — no Python loops. Swap numpy for
cupy and it runs on GPU unchanged.
"""

import numpy as np
from dataclasses import dataclass


# ═══════════════════════════════════════════════════════════
#  Parameters — 5 physics knobs + timestep
# ═══════════════════════════════════════════════════════════

@dataclass
class Params:
    """Every parameter is one physical mechanism.

    solar    — how much the sun heats the surface
    cooling  — how fast air radiates heat away
    c_sq     — how strongly temperature differences push wind
    diffuse  — how fast turbulence mixes everything
    drag     — how much friction slows the wind
    """
    dt: float = 0.4
    solar: float = 0.15
    cooling: float = 0.02
    c_sq: float = 0.15
    diffuse: float = 0.1
    drag: float = 0.02


# ═══════════════════════════════════════════════════════════
#  Spatial operators — shared by both geometries
# ═══════════════════════════════════════════════════════════

def _gradient(f):
    """Central difference. Periodic in x (cols), bounded in y (rows)."""
    dfdx = (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / 2
    dfdy = np.zeros_like(f)
    dfdy[1:-1] = (f[2:] - f[:-2]) / 2
    dfdy[0] = f[1] - f[0]
    dfdy[-1] = f[-1] - f[-2]
    return dfdx, dfdy


def _laplacian(f):
    """5-point Laplacian. Periodic in x, Neumann (zero-flux) in y."""
    lap = np.roll(f, 1, axis=1) + np.roll(f, -1, axis=1) - 2 * f
    fp = np.pad(f, ((1, 1), (0, 0)), mode='edge')
    lap += fp[2:] + fp[:-2] - 2 * f
    return lap


def _advect(field, u, v, dt):
    """Upwind advection. Wind carries field values along.

    Uses backward difference where wind blows forward and vice versa.
    First-order, monotone, unconditionally stable.
    """
    # x-direction (periodic)
    dfdx = np.where(
        u > 0,
        field - np.roll(field, 1, axis=1),   # backward
        np.roll(field, -1, axis=1) - field,   # forward
    )
    # y-direction (bounded)
    dfdy_back = np.zeros_like(field)
    dfdy_back[1:] = field[1:] - field[:-1]
    dfdy_fwd = np.zeros_like(field)
    dfdy_fwd[:-1] = field[1:] - field[:-1]
    dfdy = np.where(v > 0, dfdy_back, dfdy_fwd)

    return field - dt * (u * dfdx + v * dfdy)


# ═══════════════════════════════════════════════════════════
#  Grid geometry — lat × lon
# ═══════════════════════════════════════════════════════════

class GridGeometry:
    """Lat-lon rectangle. Periodic east-west, bounded at poles."""

    def __init__(self, n_lat, n_lon):
        self.shape = (n_lat, n_lon)
        lat = np.linspace(-np.pi / 2, np.pi / 2, n_lat)
        lon = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)
        self.lat_grid, self.lon_grid = np.meshgrid(lat, lon, indexing='ij')
        self.sin_lat = np.sin(self.lat_grid)
        self.cos_lat = np.cos(self.lat_grid)

    def solar(self, sun_lon):
        """Q ∝ max(0, cos(lat) · cos(lon − sun))"""
        return np.maximum(0.0, self.cos_lat * np.cos(self.lon_grid - sun_lon))


# ═══════════════════════════════════════════════════════════
#  Ring geometry — altitude × azimuth
# ═══════════════════════════════════════════════════════════

class RingGeometry:
    """Vertical ring cross-section. Periodic around, bounded top/bottom."""

    def __init__(self, n_layers, n_cells):
        self.shape = (n_layers, n_cells)
        self.theta = np.linspace(0, 2 * np.pi, n_cells, endpoint=False)

    def solar(self, sun_angle):
        """Sun heats ground layer only. Q ∝ max(0, cos(θ − sun))"""
        pattern = np.zeros(self.shape)
        pattern[0, :] = np.maximum(0.0, np.cos(self.theta - sun_angle))
        return pattern


# ═══════════════════════════════════════════════════════════
#  step_grid — 5 operations + Coriolis
# ═══════════════════════════════════════════════════════════

def step_grid(T, u, v, geo, p, sun_lon, omega=0.4):
    """One timestep on a lat-lon grid.

    omega — planetary rotation rate (Coriolis strength)
    """
    # 1. Heat — sun warms the day side
    T = T + p.dt * p.solar * geo.solar(sun_lon)

    # 2. Cool — radiation to space, proportional to temperature
    T = T * (1 - p.dt * p.cooling)

    # 3. Pressure — temperature gradient accelerates wind
    dTdx, dTdy = _gradient(T)
    u = u - p.dt * p.c_sq * dTdx
    v = v - p.dt * p.c_sq * dTdy

    # Coriolis — rotation deflects wind (exact rotation, conserves speed)
    f = 2 * omega * geo.sin_lat * p.dt
    c, s = np.cos(f), np.sin(f)
    u, v = u * c + v * s, -u * s + v * c

    # 4. Advect — wind carries heat and momentum
    u0, v0 = u.copy(), v.copy()
    T = _advect(T, u0, v0, p.dt)
    u = _advect(u, u0, v0, p.dt)
    v = _advect(v, u0, v0, p.dt)

    # 5. Diffuse + friction
    T = T + p.dt * p.diffuse * _laplacian(T)
    u = u + p.dt * p.diffuse * _laplacian(u)
    v = v + p.dt * p.diffuse * _laplacian(v)
    u = u * (1 - p.dt * p.drag)
    v = v * (1 - p.dt * p.drag)

    # Boundary: no flow through poles
    v[0, :] = 0
    v[-1, :] = 0

    return T, u, v


# ═══════════════════════════════════════════════════════════
#  step_ring — 5 operations + buoyancy
# ═══════════════════════════════════════════════════════════

def step_ring(T, u, w, geo, p, sun_angle, buoyancy=0.2):
    """One timestep on a vertical ring.

    buoyancy — how strongly hot air rises
    """
    # 1. Heat — sun warms the ground
    T = T + p.dt * p.solar * geo.solar(sun_angle)

    # 2. Cool — radiation to space
    T = T * (1 - p.dt * p.cooling)

    # 3. Pressure — horizontal temperature gradient → horizontal wind
    dTdx, _ = _gradient(T)
    u = u - p.dt * p.c_sq * dTdx

    # Buoyancy — air rises where it's hotter than its layer average
    T_anomaly = T - T.mean(axis=1, keepdims=True)
    w = w + p.dt * buoyancy * T_anomaly

    # 4. Advect — wind carries heat and momentum
    u0, w0 = u.copy(), w.copy()
    T = _advect(T, u0, w0, p.dt)
    u = _advect(u, u0, w0, p.dt)
    w = _advect(w, u0, w0, p.dt)

    # 5. Diffuse + friction
    T = T + p.dt * p.diffuse * _laplacian(T)
    u = u + p.dt * p.diffuse * _laplacian(u)
    w = w + p.dt * p.diffuse * _laplacian(w)
    u = u * (1 - p.dt * p.drag)
    w = w * (1 - p.dt * p.drag)

    # Boundary: no flow through ground or top
    w[0, :] = 0
    w[-1, :] = 0

    return T, u, w


# ═══════════════════════════════════════════════════════════
#  Diagnostics
# ═══════════════════════════════════════════════════════════

def diagnostics(T, u, v):
    """Energy and wind stats for monitoring stability."""
    thermal = float(np.sum(T ** 2))
    kinetic = float(np.sum(u ** 2 + v ** 2))
    return {
        'thermal': thermal,
        'kinetic': kinetic,
        'total': thermal + kinetic,
        'max_wind': float(np.sqrt(np.max(u ** 2 + v ** 2))),
        'mean_T': float(np.mean(T)),
    }
