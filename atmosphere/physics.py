# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""Atmosphere Simulation — Core Physics
=====================================

Three fields on a 2D surface:

    T  — temperature  (drives pressure forces AND stores heat)
    u  — wind along x (east / azimuthal)
    v  — wind along y (north / vertical)

Seven-stage timestep pipeline:

    1. Heat       — sun adds energy at the surface
    2. Cool       — radiation removes energy (∝ temperature)
    3. Pressure   — ∇T accelerates wind (same equation, all directions)
    4. Body force — Coriolis (grid) or Gravity (ring)
    5. Advection  — wind carries T and momentum
    6. Friction   — surface drag slows wind
    7. Diffusion  — optional turbulent mixing (Laplacian smoothing)

The ring model uses a ceiling boundary (w=0 at top layer) representing
the tropopause — a real atmospheric boundary where convection stops.
The ceiling is essential: it forces rising air to turn horizontal,
creating complete circulation cells.

Stability: bounded input, proportional removal, advection conserves,
friction dissipates → energy always converges to a finite value.

All operations are pure numpy — no Python loops.
"""

import numpy as np
from dataclasses import dataclass


# ═══════════════════════════════════════════════════════════
#  Parameters — shared physics knobs + timestep
# ═══════════════════════════════════════════════════════════

@dataclass
class Params:
    """Shared parameters for all geometries.

    solar    — how much the sun heats the surface
    cooling  — how fast air radiates heat away
    c_sq     — how strongly temperature differences push wind
    diffuse  — turbulent mixing rate (optional smoothing)
    drag     — how much friction slows the wind

    Geometry-specific forces (omega, g) are passed as function
    arguments to step_grid / step_ring, not stored here.
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
    """Vertical ring cross-section. Periodic around, bounded top/bottom.

    The top boundary (ceiling) represents the tropopause — the real
    atmospheric boundary where convection stops and air turns horizontal.

    Optional (experimental) density profile for step_ring_density:
        H_cells — scale height in layer units
        rho_min — density floor

    When H_cells is None (default), all layers have equal density.
    """

    def __init__(self, n_layers, n_cells, *, H_cells=None, rho_min=0.05):
        self.shape = (n_layers, n_cells)
        self.n_layers = n_layers
        self.n_cells = n_cells
        self.theta = np.linspace(0, 2 * np.pi, n_cells, endpoint=False)

        # Density profile (experimental — used by step_ring_density only)
        if H_cells is not None and H_cells > 0:
            self.H_cells = H_cells
            z = np.arange(n_layers, dtype=float)
            self.rho = np.maximum(np.exp(-z / H_cells), rho_min)
        else:
            self.H_cells = None
            self.rho = np.ones(n_layers)

        self.rho_2d = self.rho[:, np.newaxis]  # (n_layers, 1)

    def solar(self, sun_angle):
        """Sun heats ground layer only. Q ∝ max(0, cos(θ − sun))"""
        pattern = np.zeros(self.shape)
        pattern[0, :] = np.maximum(0.0, np.cos(self.theta - sun_angle))
        return pattern


# ═══════════════════════════════════════════════════════════
#  step_grid — pipeline with Coriolis
# ═══════════════════════════════════════════════════════════

def step_grid(T, u, v, geo, p, sun_lon, omega=0.4):
    """One timestep on a lat-lon grid.

    Pipeline: Heat → Cool → Pressure → Coriolis → Advect → Friction → Diffusion
    omega — planetary rotation rate (Coriolis strength)
    """
    # 1. Heat — sun warms the day side
    T = T + p.dt * p.solar * geo.solar(sun_lon)

    # 2. Cool — radiation to space
    T = T * (1 - p.dt * p.cooling)

    # 3. Pressure — ∇T → wind acceleration (both directions)
    dTdx, dTdy = _gradient(T)
    u = u - p.dt * p.c_sq * dTdx
    v = v - p.dt * p.c_sq * dTdy

    # 4. Coriolis — rotation deflects wind (exact rotation matrix, conserves speed)
    f = 2 * omega * geo.sin_lat * p.dt
    c, s = np.cos(f), np.sin(f)
    u, v = u * c + v * s, -u * s + v * c

    # 5. Advect — wind carries T, u, v
    u0, v0 = u.copy(), v.copy()
    T = _advect(T, u0, v0, p.dt)
    u = _advect(u, u0, v0, p.dt)
    v = _advect(v, u0, v0, p.dt)

    # 6. Friction
    u = u * (1 - p.dt * p.drag)
    v = v * (1 - p.dt * p.drag)

    # 7. Diffusion (optional smoothing)
    T = T + p.dt * p.diffuse * _laplacian(T)
    u = u + p.dt * p.diffuse * _laplacian(u)
    v = v + p.dt * p.diffuse * _laplacian(v)

    # Boundary: no flow through poles
    v[0, :] = 0
    v[-1, :] = 0

    return T, u, v


# ═══════════════════════════════════════════════════════════
#  step_ring — symmetric pressure + gravity
# ═══════════════════════════════════════════════════════════

def step_ring(T, u, w, geo, p, sun_angle, g=0.08, *,
              diffuse_T=None, diffuse_wind=None):
    """One timestep on a vertical ring.

    Pipeline: Heat → Cool → Pressure → Gravity → Advect → Friction → Diffusion

    g — gravitational acceleration (constant downward pull on all air)

    Pressure is applied symmetrically in both directions (same equation
    for horizontal and vertical). Gravity provides the restoring force.
    Buoyancy-like circulation emerges from their competition: where the
    ground is hot, the upward pressure gradient exceeds gravity and air
    rises; where it's cold, gravity wins and air sinks.

    Optional overrides for analysis:
        diffuse_T    — diffusion coefficient for temperature only
        diffuse_wind — diffusion coefficient for wind fields only
        When None, both use p.diffuse.
    """
    dT = p.diffuse if diffuse_T is None else diffuse_T
    dW = p.diffuse if diffuse_wind is None else diffuse_wind

    # 1. Heat — sun warms the ground
    T = T + p.dt * p.solar * geo.solar(sun_angle)

    # 2. Cool — radiation to space
    T = T * (1 - p.dt * p.cooling)

    # 3. Pressure — ∇T → wind acceleration (same equation both directions)
    dTdx, dTdz = _gradient(T)
    u = u - p.dt * p.c_sq * dTdx
    w = w - p.dt * p.c_sq * dTdz

    # 4. Gravity — constant downward pull
    w = w - p.dt * g

    # 5. Advect — wind carries T, u, w
    u0, w0 = u.copy(), w.copy()
    T = _advect(T, u0, w0, p.dt)
    u = _advect(u, u0, w0, p.dt)
    w = _advect(w, u0, w0, p.dt)

    # 6. Friction
    u = u * (1 - p.dt * p.drag)
    w = w * (1 - p.dt * p.drag)

    # 7. Diffusion (optional smoothing)
    if dT > 0:
        T = T + p.dt * dT * _laplacian(T)
    if dW > 0:
        u = u + p.dt * dW * _laplacian(u)
        w = w + p.dt * dW * _laplacian(w)

    # Boundary: no flow through ground or top
    w[0, :] = 0
    w[-1, :] = 0

    return T, u, w


# ═══════════════════════════════════════════════════════════
#  step_ring_legacy — anomaly buoyancy (for comparison)
# ═══════════════════════════════════════════════════════════

def step_ring_legacy(T, u, w, geo, p, sun_angle, buoyancy=0.2):
    """Ring step using horizontal anomaly buoyancy (original model).

    Kept for sensitivity analysis comparison. The canonical model is
    step_ring() which uses symmetric pressure + gravity instead.

    buoyancy — how strongly hot-vs-layer-average temperature drives w
    """
    # 1. Heat
    T = T + p.dt * p.solar * geo.solar(sun_angle)

    # 2. Cool
    T = T * (1 - p.dt * p.cooling)

    # 3. Pressure — horizontal only
    dTdx, _ = _gradient(T)
    u = u - p.dt * p.c_sq * dTdx

    # 4. Buoyancy — air rises where hotter than layer average
    T_anomaly = T - T.mean(axis=1, keepdims=True)
    w = w + p.dt * buoyancy * T_anomaly

    # 5. Advect
    u0, w0 = u.copy(), w.copy()
    T = _advect(T, u0, w0, p.dt)
    u = _advect(u, u0, w0, p.dt)
    w = _advect(w, u0, w0, p.dt)

    # 6. Friction + 7. Diffusion
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
#  step_ring_pressure_only — symmetric pressure, no gravity
#  (analysis variant: shows net-upward bias without gravity)
# ═══════════════════════════════════════════════════════════

def step_ring_pressure_only(T, u, w, geo, p, sun_angle):
    """Ring step using symmetric 2D pressure but no gravity.

    Analysis-only variant. Demonstrates the net-upward bias that occurs
    when vertical pressure gradient has no restoring force: hot ground
    always pushes up, nothing pulls down.
    """
    return step_ring(T, u, w, geo, p, sun_angle, g=0.0)


# ═══════════════════════════════════════════════════════════
#  step_ring_density — EXPERIMENTAL, does not produce circulation
# ═══════════════════════════════════════════════════════════

def step_ring_density(T, u, w, geo, p, sun_angle, g=0.08, *,
                      diffuse_T=None, diffuse_wind=None):
    """EXPERIMENTAL: Ring step with layer density (does NOT yet work).

    Same 7-stage pipeline as step_ring, but vertical pressure uses the
    density profile from geo.rho:

        Vertical pressure: w -= (c_sq/ρ) × ∂(ρT)/∂z

    Status: produces net-upward bias without circulation cells.
    The density gradient adds to the upward pressure force, and gravity
    alone cannot restore the balance needed for realistic circulation.
    Kept for future investigation — see docs/SUMMARY.md.

    Requires geo to have H_cells set (creates exponential density profile).
    With uniform density (H_cells=None), this is identical to step_ring.
    """
    dT_coeff = p.diffuse if diffuse_T is None else diffuse_T
    dW_coeff = p.diffuse if diffuse_wind is None else diffuse_wind
    rho = geo.rho_2d  # (n_layers, 1) — broadcasts over cells

    # 1. Heat — sun warms the ground
    T = T + p.dt * p.solar * geo.solar(sun_angle)

    # 2. Cool — radiation to space
    T = T * (1 - p.dt * p.cooling)

    # 3. Pressure
    #    Horizontal: ρ is constant within a layer, cancels out
    dTdx, _ = _gradient(T)
    u = u - p.dt * p.c_sq * dTdx

    #    Vertical: pressure = ρ × T, force per unit mass = -(c_sq/ρ) × ∂(ρT)/∂z
    P = rho * T
    _, dPdz = _gradient(P)
    w = w - p.dt * (p.c_sq / rho) * dPdz

    # 4. Gravity — constant downward pull (same as step_ring)
    w = w - p.dt * g

    # 5. Advect — wind carries T, u, w
    u0, w0 = u.copy(), w.copy()
    T = _advect(T, u0, w0, p.dt)
    u = _advect(u, u0, w0, p.dt)
    w = _advect(w, u0, w0, p.dt)

    # 6. Friction
    u = u * (1 - p.dt * p.drag)
    w = w * (1 - p.dt * p.drag)

    # 7. Diffusion (optional smoothing)
    if dT_coeff > 0:
        T = T + p.dt * dT_coeff * _laplacian(T)
    if dW_coeff > 0:
        u = u + p.dt * dW_coeff * _laplacian(u)
        w = w + p.dt * dW_coeff * _laplacian(w)

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
    max_speed = float(np.sqrt(np.max(u ** 2 + v ** 2)))
    return {
        'thermal': thermal,
        'kinetic': kinetic,
        'total': thermal + kinetic,
        'max_wind': max_speed,
        'mean_T': float(np.mean(T)),
        'courant': max_speed,   # CFL number ≈ max_speed * dt (dx=1)
    }


def ring_diagnostics(T, u, w):
    """Extended diagnostics for ring geometry.

    Extra fields beyond diagnostics():
        vert_ke       — kinetic energy in vertical wind
        horiz_ke      — kinetic energy in horizontal wind
        net_w         — mean vertical velocity (≈0 for balanced cells)
        heat_transport — mean(w·T): positive = warm air rising
        lapse_rate    — mean vertical temperature gradient
    """
    base = diagnostics(T, u, w)
    base['vert_ke']        = float(np.sum(w ** 2))
    base['horiz_ke']       = float(np.sum(u ** 2))
    base['net_w']          = float(np.mean(w))
    base['heat_transport'] = float(np.mean(w * T))
    if T.shape[0] > 1:
        dTdz = np.mean(T[-1] - T[0]) / (T.shape[0] - 1)
        base['lapse_rate'] = float(dTdz)
    return base
