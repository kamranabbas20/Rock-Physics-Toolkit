"""Diagnostic rock-physics models: bounds, trends and dry-frame templates.

These are *model overlays* to diagnose log data against — mixture bounds,
empirical velocity trends, and granular-medium frame models.  Nothing here
substitutes fluids in the input well: there is no Gassmann and no
Batzle-Wang, per SPEC.md section 1.  The Hashin-Shtrikman and Voigt-Reuss-Hill
bounds are computed on a mineral-plus-fluid mixture, so they bracket the
*saturated* rock directly without any substitution step; the granular models
(Hertz-Mindlin, soft and stiff sand) describe the **dry frame** and are
labelled as such wherever they are used.

Unit conventions: velocities m/s, densities g/cc, elastic moduli GPa,
porosity and mineral fractions as fractions (v/v), pressure Pa.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "MINERALS",
    "FLUIDS",
    "bulk_modulus",
    "shear_modulus",
    "velocities_from_moduli",
    "voigt",
    "reuss",
    "voigt_reuss_hill",
    "hashin_shtrikman",
    "critical_porosity_dry",
    "coordination_number",
    "hertz_mindlin",
    "soft_sand_dry",
    "stiff_sand_dry",
    "wyllie",
    "raymer_hunt_gardner",
    "gardner_density",
    "gardner_velocity",
    "fit_gardner",
    "castagna_mudrock",
    "greenberg_castagna",
]

#: Mineral end members: bulk modulus (GPa), shear modulus (GPa), density (g/cc).
MINERALS = {
    "quartz": (36.6, 45.0, 2.65),
    "clay": (21.0, 7.0, 2.58),
    "calcite": (76.8, 32.0, 2.71),
    "dolomite": (94.9, 45.0, 2.87),
    "feldspar": (75.6, 25.6, 2.63),
}

#: Pore fluids: bulk modulus (GPa), density (g/cc).  Representative values —
#: a real study would compute these for its own P, T and salinity.
FLUIDS = {
    "brine": (2.80, 1.09),
    "oil": (0.94, 0.78),
    "gas": (0.021, 0.25),
}


def _arr(x):
    return np.asarray(x, dtype=float)


# ------------------------------------------------- moduli from velocities ---
def bulk_modulus(vp, vs, rho):
    """Bulk modulus K in GPa from Vp (m/s), Vs (m/s) and density (g/cc)."""
    vp, vs, rho = _arr(vp), _arr(vs), _arr(rho)
    return rho * (vp ** 2 - 4.0 / 3.0 * vs ** 2) / 1e6


def shear_modulus(vs, rho):
    """Shear modulus G in GPa from Vs (m/s) and density (g/cc)."""
    return _arr(rho) * _arr(vs) ** 2 / 1e6


def velocities_from_moduli(K, G, rho):
    """Inverse of the two above: ``(vp, vs)`` in m/s from K, G (GPa), rho (g/cc)."""
    K, G, rho = _arr(K), _arr(G), _arr(rho)
    with np.errstate(divide="ignore", invalid="ignore"):
        vp = np.sqrt((K + 4.0 / 3.0 * G) * 1e6 / rho)
        vs = np.sqrt(G * 1e6 / rho)
    return np.where(np.isfinite(vp), vp, np.nan), np.where(np.isfinite(vs), vs, np.nan)


# ------------------------------------------------------------- bounds -------
def _mix_inputs(moduli, fractions):
    m = np.atleast_1d(_arr(moduli))
    f = np.atleast_1d(_arr(fractions))
    if m.shape != f.shape:
        raise ValueError("moduli and fractions must have the same shape")
    total = f.sum()
    if total <= 0:
        raise ValueError("mineral fractions must sum to a positive number")
    return m, f / total


def voigt(moduli, fractions):
    """Voigt (iso-strain) upper bound of a mixture."""
    m, f = _mix_inputs(moduli, fractions)
    return float(np.sum(f * m))


def reuss(moduli, fractions):
    """Reuss (iso-stress) lower bound of a mixture.

    A zero-modulus phase (a fluid's shear modulus) drives the bound to zero,
    which is the physically correct answer rather than an error.
    """
    m, f = _mix_inputs(moduli, fractions)
    if np.any(m <= 0):
        return 0.0
    return float(1.0 / np.sum(f / m))


def voigt_reuss_hill(moduli, fractions):
    """Hill average of the Voigt and Reuss bounds."""
    return 0.5 * (voigt(moduli, fractions) + reuss(moduli, fractions))


def hashin_shtrikman(K1, G1, K2, G2, f1):
    """Hashin-Shtrikman bounds for a two-phase mixture.

    Parameters
    ----------
    K1, G1 : float
        Moduli of phase 1 (GPa), present at volume fraction ``f1``.
    K2, G2 : float
        Moduli of phase 2 (GPa).
    f1 : array_like
        Volume fraction of phase 1.

    Returns
    -------
    dict
        ``{'K_upper', 'K_lower', 'G_upper', 'G_lower'}``, each an array over
        ``f1``.  The upper bound puts the stiffer phase in the shell, the
        lower bound the softer one; the two coincide for a single phase.

    Notes
    -----
    A fluid phase (G = 0) makes the shear lower bound identically zero, which
    is handled explicitly rather than by dividing by zero.
    """
    f1 = np.atleast_1d(_arr(f1))
    f2 = 1.0 - f1

    def _hs(Ka, Ga, Kb, Gb, fa, fb):
        """Mixture with phase a as the enveloping shell."""
        with np.errstate(divide="ignore", invalid="ignore"):
            K = Ka + fb / (1.0 / (Kb - Ka) + fa / (Ka + 4.0 / 3.0 * Ga))
            if Ga <= 0:
                G = np.zeros_like(fb)
            else:
                zeta = Ga / 6.0 * (9.0 * Ka + 8.0 * Ga) / (Ka + 2.0 * Ga)
                G = Ga + fb / (1.0 / (Gb - Ga) + fa / (Ga + zeta))
        return K, G

    # Degenerate case: identical phases, or a fraction pinned at 0 or 1.
    if np.isclose(K1, K2) and np.isclose(G1, G2):
        K = np.full(f1.shape, float(K1))
        G = np.full(f1.shape, float(G1))
        return {"K_upper": K, "K_lower": K.copy(), "G_upper": G, "G_lower": G.copy()}

    Ka, Ga = _hs(K1, G1, K2, G2, f1, f2)
    Kb, Gb = _hs(K2, G2, K1, G1, f2, f1)

    stack_k = np.vstack([Ka, Kb])
    stack_g = np.vstack([Ga, Gb])
    out = {
        "K_upper": np.nanmax(stack_k, axis=0),
        "K_lower": np.nanmin(stack_k, axis=0),
        "G_upper": np.nanmax(stack_g, axis=0),
        "G_lower": np.nanmin(stack_g, axis=0),
    }
    # At the end members both bounds collapse onto the pure phase.
    for i, (frac, pure) in enumerate(((f1, (K1, G1)), (f2, (K2, G2)))):
        at_end = np.isclose(frac, 1.0)
        if at_end.any():
            out["K_upper"][at_end] = out["K_lower"][at_end] = pure[0]
            out["G_upper"][at_end] = out["G_lower"][at_end] = pure[1]
    return out


# ----------------------------------------------------- frame models ---------
def critical_porosity_dry(K_mineral, G_mineral, phi, phi_c=0.40):
    """Nur's critical-porosity (modified Voigt) **dry** frame.

    Moduli fall linearly from the mineral value at zero porosity to zero at
    the critical porosity, above which the rock is a suspension.
    """
    phi = _arr(phi)
    ratio = np.clip(phi / float(phi_c), 0.0, 1.0)
    return float(K_mineral) * (1.0 - ratio), float(G_mineral) * (1.0 - ratio)


def coordination_number(phi):
    """Murphy's (1982) empirical grain coordination number."""
    phi = _arr(phi)
    return 20.0 - 34.0 * phi + 14.0 * phi ** 2


def hertz_mindlin(K_mineral, G_mineral, phi_c=0.36, n=None, pressure=10e6, f=1.0):
    """Hertz-Mindlin **dry** moduli of a random pack at the critical porosity.

    Parameters
    ----------
    K_mineral, G_mineral : float
        Mineral moduli in GPa.
    phi_c : float
        Critical porosity of the pack.
    n : float, optional
        Coordination number; defaults to Murphy's relation at ``phi_c``.
    pressure : float
        Effective pressure in Pa.
    f : float
        Shear (tangential-stiffness) factor: 1 for perfect adhesion, 0 for
        frictionless grains.

    Returns
    -------
    (K_hm, G_hm) : tuple of float
        Dry-frame moduli in GPa.
    """
    if n is None:
        n = float(coordination_number(phi_c))
    K_pa = float(K_mineral) * 1e9
    G_pa = float(G_mineral) * 1e9
    nu = (3.0 * K_pa - 2.0 * G_pa) / (2.0 * (3.0 * K_pa + G_pa))

    K_hm = (
        n ** 2 * (1.0 - phi_c) ** 2 * G_pa ** 2 * float(pressure)
        / (18.0 * np.pi ** 2 * (1.0 - nu) ** 2)
    ) ** (1.0 / 3.0)
    G_hm = (2.0 + 3.0 * f - nu * (1.0 + 3.0 * f)) / (5.0 * (2.0 - nu)) * (
        3.0 * n ** 2 * (1.0 - phi_c) ** 2 * G_pa ** 2 * float(pressure)
        / (2.0 * np.pi ** 2 * (1.0 - nu) ** 2)
    ) ** (1.0 / 3.0)
    return K_hm / 1e9, G_hm / 1e9


def _modified_hs(K_end, G_end, K_mineral, G_mineral, phi, phi_c, z_from):
    """Modified Hashin-Shtrikman interpolation between a pack and the mineral."""
    phi = _arr(phi)
    ratio = np.clip(phi / float(phi_c), 0.0, 1.0)
    Kz, Gz = z_from
    zeta = Gz / 6.0 * (9.0 * Kz + 8.0 * Gz) / (Kz + 2.0 * Gz)
    with np.errstate(divide="ignore", invalid="ignore"):
        K = 1.0 / (
            ratio / (K_end + 4.0 / 3.0 * Gz) + (1.0 - ratio) / (K_mineral + 4.0 / 3.0 * Gz)
        ) - 4.0 / 3.0 * Gz
        G = 1.0 / (ratio / (G_end + zeta) + (1.0 - ratio) / (G_mineral + zeta)) - zeta
    return np.maximum(K, 0.0), np.maximum(G, 0.0)


def soft_sand_dry(K_mineral, G_mineral, phi, phi_c=0.36, n=None, pressure=10e6, f=1.0):
    """Dvorkin-Nur soft-sand (friable, unconsolidated) **dry** frame.

    The modified Hashin-Shtrikman *lower* bound between the Hertz-Mindlin
    pack at ``phi_c`` and the mineral point at zero porosity.
    """
    K_hm, G_hm = hertz_mindlin(K_mineral, G_mineral, phi_c, n, pressure, f)
    return _modified_hs(K_hm, G_hm, K_mineral, G_mineral, phi, phi_c, (K_hm, G_hm))


def stiff_sand_dry(K_mineral, G_mineral, phi, phi_c=0.36, n=None, pressure=10e6, f=1.0):
    """Dvorkin-Nur stiff-sand **dry** frame.

    The modified Hashin-Shtrikman *upper* bound between the same two end
    points, appropriate for a consolidated or cemented sand.
    """
    K_hm, G_hm = hertz_mindlin(K_mineral, G_mineral, phi_c, n, pressure, f)
    return _modified_hs(K_hm, G_hm, K_mineral, G_mineral, phi, phi_c,
                        (float(K_mineral), float(G_mineral)))


# --------------------------------------------- velocity-porosity trends -----
def wyllie(phi, v_matrix, v_fluid):
    """Wyllie time-average velocity-porosity relation (m/s)."""
    phi = np.clip(_arr(phi), 0.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        slowness = (1.0 - phi) / float(v_matrix) + phi / float(v_fluid)
        v = 1.0 / slowness
    return np.where(np.isfinite(v), v, np.nan)


def raymer_hunt_gardner(phi, v_matrix, v_fluid):
    """Raymer-Hunt-Gardner velocity-porosity relation (m/s), low-porosity form."""
    phi = np.clip(_arr(phi), 0.0, 1.0)
    return (1.0 - phi) ** 2 * float(v_matrix) + phi * float(v_fluid)


def gardner_density(vp, a=0.31, b=0.25):
    """Gardner's relation: density (g/cc) from Vp (m/s)."""
    return float(a) * _arr(vp) ** float(b)


def gardner_velocity(rho, a=0.31, b=0.25):
    """Gardner inverted: Vp (m/s) from density (g/cc)."""
    return (_arr(rho) / float(a)) ** (1.0 / float(b))


def fit_gardner(vp, rho):
    """Least-squares fit of ``rho = a * vp**b`` to a log, in log-log space."""
    vp, rho = _arr(vp), _arr(rho)
    good = np.isfinite(vp) & np.isfinite(rho) & (vp > 0) & (rho > 0)
    if good.sum() < 2:
        return float("nan"), float("nan")
    b, log_a = np.polyfit(np.log(vp[good]), np.log(rho[good]), 1)
    return float(np.exp(log_a)), float(b)


# -------------------------------------------------------- Vp-Vs trends ------
def castagna_mudrock(vp):
    """Castagna's mudrock line: Vs = 0.8621 Vp - 1172.4, both in m/s."""
    return 0.8621 * _arr(vp) - 1172.4


#: Greenberg-Castagna polynomial coefficients, highest power first, for Vp and
#: Vs in km/s.
_GC_COEFFS = {
    "sandstone": [0.80416, -0.85588],
    "shale": [0.76969, -0.86735],
    "limestone": [-0.05508, 1.01677, -1.03049],
    "dolomite": [0.58321, -0.07775],
}


def greenberg_castagna(vp, fractions=None):
    """Greenberg-Castagna Vs prediction for a lithology mixture.

    Parameters
    ----------
    vp : array_like
        P velocity in m/s.
    fractions : dict, optional
        ``{lithology: volume fraction}`` over ``sandstone``, ``shale``,
        ``limestone`` and ``dolomite``.  Defaults to pure sandstone.

    Returns
    -------
    ndarray
        Vs in m/s, the Hill average of the arithmetic and harmonic means of
        the single-lithology predictions.
    """
    vp = _arr(vp)
    fractions = fractions or {"sandstone": 1.0}
    unknown = set(fractions) - set(_GC_COEFFS)
    if unknown:
        raise ValueError(f"unknown lithology/-ies: {sorted(unknown)}")

    total = float(sum(fractions.values()))
    if total <= 0:
        raise ValueError("lithology fractions must sum to a positive number")

    vp_km = vp / 1000.0
    arithmetic = np.zeros_like(vp_km)
    harmonic = np.zeros_like(vp_km)
    for lith, frac in fractions.items():
        f = float(frac) / total
        vs_km = np.polyval(_GC_COEFFS[lith], vp_km)
        vs_km = np.maximum(vs_km, 1e-6)
        arithmetic += f * vs_km
        harmonic += f / vs_km

    with np.errstate(divide="ignore", invalid="ignore"):
        harmonic = 1.0 / harmonic
    return 0.5 * (arithmetic + harmonic) * 1000.0
