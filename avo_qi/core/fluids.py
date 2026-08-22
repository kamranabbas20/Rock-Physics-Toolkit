"""Pore-fluid properties at reservoir conditions — Batzle & Wang (1992).

:data:`avo_qi.core.rockphysics.FLUIDS` is a fixed table: brine at 2.80 GPa, gas
at 0.021 GPa, and so on.  Those numbers are fine for drawing a bound, but they
are wrong for substituting a real reservoir — gas at 30 MPa and 90 C is roughly
five times stiffer and six times denser than the table's entry, and a
substitution built on the table would be confidently wrong rather than
approximately right.

This module computes K and rho for brine, gas and oil from pressure,
temperature, salinity, gas gravity, GOR and API, after Batzle & Wang,
*Geophysics* 57, 1396-1408 (1992).

Conventions throughout: **pressure in MPa**, **temperature in degrees Celsius**,
**salinity as a weight fraction** (35 000 ppm is 0.035), **GOR in litres of gas
per litre of oil**, gas gravity relative to air, oil gravity in degrees API.
Results are **K in GPa** and **rho in g/cc**, matching the rest of the toolkit.

The correlations are empirical fits with a stated range — roughly 5-100 MPa and
10-350 C.  Outside it they still return numbers, so
:func:`fluid_properties` reports whether the request was inside the fitted range
rather than leaving the caller to assume it was.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "brine_properties",
    "gas_properties",
    "oil_properties",
    "fluid_properties",
    "hydrostatic_pressure",
    "geothermal_temperature",
    "in_range",
]

#: Batzle & Wang's coefficient matrix for the velocity of pure water, w[i][j]
#: multiplying ``T**i * P**j``.
_W = np.array([
    [1402.85,   1.524,     3.437e-3,  -1.197e-5],
    [4.871,    -0.0111,    1.739e-4,  -1.628e-6],
    [-0.04783,  2.747e-4, -2.135e-6,   1.237e-8],
    [1.487e-4, -6.503e-7, -1.455e-8,   1.327e-10],
    [-2.197e-7, 7.987e-10, 5.230e-11, -4.614e-13],
])

#: The range the correlations were fitted over (MPa, degrees C).
FITTED_RANGE = {"pressure": (5.0, 100.0), "temperature": (10.0, 350.0)}


def in_range(pressure, temperature):
    """True where ``pressure`` and ``temperature`` sit inside the fitted range."""
    p, t = np.asarray(pressure, float), np.asarray(temperature, float)
    p_lo, p_hi = FITTED_RANGE["pressure"]
    t_lo, t_hi = FITTED_RANGE["temperature"]
    return (p >= p_lo) & (p <= p_hi) & (t >= t_lo) & (t <= t_hi)


# ---------------------------------------------------------- depth helpers ---
def hydrostatic_pressure(depth, gradient=10.0, datum=0.0):
    """Pore pressure in MPa from depth in metres.

    ``gradient`` is in MPa/km — 10.0 is a normally pressured column of brine.
    Overpressure is expressed by raising it.
    """
    return (np.asarray(depth, float) - float(datum)) * float(gradient) / 1000.0


def geothermal_temperature(depth, surface=15.0, gradient=30.0, datum=0.0):
    """Temperature in degrees C from depth in metres.

    ``gradient`` is in degrees C per kilometre.
    """
    return float(surface) + (np.asarray(depth, float) - float(datum)) \
        * float(gradient) / 1000.0


# ------------------------------------------------------------------ brine ---
def _water_density(pressure, temperature):
    p, t = pressure, temperature
    return 1.0 + 1e-6 * (
        -80.0 * t - 3.3 * t ** 2 + 0.00175 * t ** 3
        + 489.0 * p - 2.0 * t * p + 0.016 * t ** 2 * p - 1.3e-5 * t ** 3 * p
        - 0.333 * p ** 2 - 0.002 * t * p ** 2
    )


def _water_velocity(pressure, temperature):
    p, t = pressure, temperature
    v = np.zeros(np.broadcast(np.asarray(p), np.asarray(t)).shape, dtype=float)
    for i in range(_W.shape[0]):
        for j in range(_W.shape[1]):
            v = v + _W[i, j] * t ** i * p ** j
    return v


def brine_properties(pressure, temperature, salinity=0.035):
    """Bulk modulus (GPa) and density (g/cc) of brine.

    ``salinity`` is a weight fraction of sodium chloride equivalent, so
    seawater is about 0.035 and a strong formation brine 0.15 or more.
    """
    p = np.asarray(pressure, dtype=float)
    t = np.asarray(temperature, dtype=float)
    s = np.asarray(salinity, dtype=float)

    rho_w = _water_density(p, t)
    rho = rho_w + s * (
        0.668 + 0.44 * s + 1e-6 * (
            300.0 * p - 2400.0 * p * s
            + t * (80.0 + 3.0 * t - 3300.0 * s - 13.0 * p + 47.0 * p * s)
        )
    )

    v = _water_velocity(p, t) + s * (
        1170.0 - 9.6 * t + 0.055 * t ** 2 - 8.5e-5 * t ** 3
        + 2.6 * p - 0.0029 * t * p - 0.0476 * p ** 2
    ) + s ** 1.5 * (780.0 - 10.0 * p + 0.16 * p ** 2) - 1820.0 * s ** 2

    return _modulus(rho, v), rho


# -------------------------------------------------------------------- gas ---
def gas_properties(pressure, temperature, gravity=0.6):
    """Bulk modulus (GPa) and density (g/cc) of a hydrocarbon gas.

    ``gravity`` is the molecular weight relative to air: about 0.56 for dry
    methane, rising towards 1.0 and beyond as heavier components come in.

    The modulus is the **adiabatic** one, which is what a seismic wave sees.
    """
    p = np.asarray(pressure, dtype=float)
    t = np.asarray(temperature, dtype=float)
    g = np.asarray(gravity, dtype=float)
    t_abs = t + 273.15

    p_pr = p / (4.892 - 0.4048 * g)
    t_pr = t_abs / (94.72 + 170.75 * g)

    # Compressibility factor, and the part of it that depends on pressure.
    exponent = -(0.45 + 8.0 * (0.56 - 1.0 / t_pr) ** 2) * p_pr ** 1.2 / t_pr
    e = 0.109 * (3.85 - t_pr) ** 2 * np.exp(exponent)
    a = 0.03 + 0.00527 * (3.5 - t_pr) ** 3
    z = a * p_pr + (0.642 * t_pr - 0.007 * t_pr ** 4 - 0.52) + e

    # dE/dPpr, and hence dZ/dPpr.
    de = e * (-1.2 * p_pr ** 0.2 * (0.45 + 8.0 * (0.56 - 1.0 / t_pr) ** 2) / t_pr)
    dz = a + de

    gamma = (0.85 + 5.6 / (p_pr + 2.0) + 27.1 / (p_pr + 3.5) ** 2
             - 8.7 * np.exp(-0.65 * (p_pr + 1.0)))

    # 28.8 g/mol is air; R in cm^3 MPa / (mol K) makes this come out in g/cc.
    rho = 28.8 * g * p / (z * 8.314 * t_abs)

    with np.errstate(divide="ignore", invalid="ignore"):
        k_mpa = p * gamma / (1.0 - p_pr / z * dz)
    return k_mpa / 1000.0, rho


# -------------------------------------------------------------------- oil ---
def oil_properties(pressure, temperature, api=30.0, gor=0.0, gas_gravity=0.6):
    """Bulk modulus (GPa) and density (g/cc) of oil.

    ``gor`` is the gas-oil ratio in litres per litre.  Leave it at zero for a
    dead oil; a non-zero value applies Batzle & Wang's live-oil correction,
    which softens and lightens the oil considerably — a GOR of 100 L/L can
    halve the modulus, so it is not a detail worth defaulting away.
    """
    p = np.asarray(pressure, dtype=float)
    t = np.asarray(temperature, dtype=float)
    gor = np.asarray(gor, dtype=float)
    g = np.asarray(gas_gravity, dtype=float)

    rho_0 = 141.5 / (float(api) + 131.5)          # reference density at 15.6 C

    if np.all(gor <= 0.0):
        rho_ref = rho_0
        rho_pseudo = rho_0
    else:
        # Volume factor of the oil once the dissolved gas is in it.
        b_0 = 0.972 + 0.00038 * (
            2.4 * gor * np.sqrt(g / rho_0) + t + 17.8) ** 1.175
        # True density carries the dissolved gas's mass; the velocity
        # correlation instead wants the gas-free "pseudo" density.
        rho_ref = (rho_0 + 0.0012 * g * gor) / b_0
        rho_pseudo = rho_0 / b_0 / (1.0 + 0.001 * gor)

    # Pressure and temperature correction, applied to whichever density
    # the stage needs.
    def _corrected(rho_r):
        rho_p = (rho_r + (0.00277 * p - 1.71e-7 * p ** 3) * (rho_r - 1.15) ** 2
                 + 3.49e-4 * p)
        return rho_p / (0.972 + 3.81e-4 * (t + 17.78) ** 1.175)

    rho = _corrected(rho_ref)

    v = (2096.0 * np.sqrt(rho_pseudo / (2.6 - rho_pseudo))
         - 3.7 * t + 4.64 * p
         + 0.0115 * (4.12 * np.sqrt(1.08 / rho_pseudo - 1.0) - 1.0) * t * p)

    return _modulus(rho, v), rho


def _modulus(rho, velocity):
    """``K = rho * V**2`` in GPa, for rho in g/cc and V in m/s."""
    return np.asarray(rho, float) * np.asarray(velocity, float) ** 2 * 1e-6


# ----------------------------------------------------------------- facade ---
def fluid_properties(fluid, pressure, temperature, salinity=0.035, api=30.0,
                     gor=0.0, gas_gravity=0.6):
    """``(K, rho, warning)`` for ``'brine'``, ``'oil'`` or ``'gas'``.

    ``warning`` is a sentence when the conditions fall outside the range the
    correlations were fitted over, and ``None`` when they do not — so a caller
    that shows it cannot silently present an extrapolation as a measurement.
    """
    key = str(fluid).strip().lower()
    if key == "brine":
        k, rho = brine_properties(pressure, temperature, salinity)
    elif key == "gas":
        k, rho = gas_properties(pressure, temperature, gas_gravity)
    elif key == "oil":
        k, rho = oil_properties(pressure, temperature, api, gor, gas_gravity)
    else:
        raise ValueError(
            f"unknown fluid {fluid!r}; expected 'brine', 'oil' or 'gas'")

    ok = in_range(pressure, temperature)
    warning = None
    if not np.all(ok):
        p_lo, p_hi = FITTED_RANGE["pressure"]
        t_lo, t_hi = FITTED_RANGE["temperature"]
        warning = (
            f"{np.size(ok) - int(np.sum(ok))} of {np.size(ok)} condition(s) fall "
            f"outside the range Batzle-Wang was fitted over "
            f"({p_lo:.0f}-{p_hi:.0f} MPa, {t_lo:.0f}-{t_hi:.0f} C); "
            "the values are extrapolations."
        )
    return float(k) if np.ndim(k) == 0 else k, \
        float(rho) if np.ndim(rho) == 0 else rho, warning
