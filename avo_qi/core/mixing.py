"""Mixing laws for minerals and for pore fluids.

Two different problems wear the same word.

**Fluid mixing** asks what a pore filled with brine, oil and gas behaves like.
The answer depends on how the phases are *arranged*, not just how much of each
there is.  Finely mixed at the pore scale, the fluids share a common pressure
and the bulk moduli average harmonically — Wood's law.  Segregated into
patches larger than the diffusion length, each patch stiffens independently
and the average is arithmetic — the patchy limit.  Real rocks sit between, and
Brie's empirical curve is the usual way to park a saturation there.  The
difference is large: a few percent gas drops a brine-filled modulus by an
order of magnitude under Wood, and barely moves it under the patchy limit.

**Mineral mixing** asks what a matrix of quartz, clay and carbonate behaves
like.  Bounds are the honest answer — Voigt and Reuss at the outside, the
narrower Hashin-Shtrikman-Walpole pair inside — because the geometry that
would pin down a single value is not known.

Moduli in GPa, densities in g/cc, saturations and fractions as fractions.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "FLUID_MIXING_LAWS",
    "MINERAL_MIXING_LAWS",
    "wood",
    "patchy",
    "brie",
    "fluid_mix",
    "fluid_density",
    "hashin_shtrikman_walpole",
    "mineral_mix",
]


def _normalised(fractions, what="fractions"):
    f = np.atleast_1d(np.asarray(fractions, dtype=float))
    if np.any(f < 0):
        raise ValueError(f"{what} must not be negative")
    total = f.sum()
    if total <= 0:
        raise ValueError(f"{what} must sum to a positive number")
    return f / total


# ------------------------------------------------------------- fluids ------
def wood(moduli, saturations):
    """Wood's law: the harmonic average, for fluids finely mixed in the pore.

    The soft-fluid limit — a little gas dominates, because the phases share a
    pressure and the most compressible one gives way first.
    """
    k = np.atleast_1d(np.asarray(moduli, dtype=float))
    s = _normalised(saturations, "saturations")
    if k.shape != s.shape:
        raise ValueError("moduli and saturations must have the same shape")
    if np.any(k <= 0):
        return 0.0
    return float(1.0 / np.sum(s / k))


def patchy(moduli, saturations):
    """The patchy (Voigt) limit: arithmetic averaging.

    Appropriate where each fluid occupies patches larger than the pressure
    diffusion length, so they stiffen independently rather than sharing a
    pressure.  The stiff-fluid limit.
    """
    k = np.atleast_1d(np.asarray(moduli, dtype=float))
    s = _normalised(saturations, "saturations")
    if k.shape != s.shape:
        raise ValueError("moduli and saturations must have the same shape")
    return float(np.sum(s * k))


def brie(k_liquid, k_gas, s_liquid, exponent=3.0):
    """Brie's empirical mixing curve between the two limits.

    ``K = (K_liquid - K_gas) * S_liquid**e + K_gas``.  ``e = 1`` reproduces
    the patchy (arithmetic) limit; larger exponents bend towards Wood, with
    ``e = 3`` the common default for a gas-brine mix.

    It is an empirical fit rather than a bound: above roughly 90% gas the
    ``e = 3`` curve crosses marginally *below* Wood's.  The difference there is
    a fraction of a percent of the gas modulus and of no practical
    consequence, but it is real.
    """
    k_liquid, k_gas = float(k_liquid), float(k_gas)
    exponent = float(exponent)
    if exponent <= 0:
        raise ValueError("the Brie exponent must be positive")
    s = np.clip(np.asarray(s_liquid, dtype=float), 0.0, 1.0)
    out = (k_liquid - k_gas) * s ** exponent + k_gas
    return float(out) if np.isscalar(s_liquid) or out.ndim == 0 else out


def hill_fluid(moduli, saturations):
    """The average of Wood and the patchy limit — a blunt midpoint."""
    return 0.5 * (wood(moduli, saturations) + patchy(moduli, saturations))


#: Fluid mixing laws, keyed by the name the UI shows.
FLUID_MIXING_LAWS = {
    "wood": wood,
    "patchy": patchy,
    "hill": hill_fluid,
}


def fluid_mix(moduli, saturations, method="wood", brie_exponent=3.0,
              gas_index=None):
    """Effective pore-fluid bulk modulus under the chosen mixing law.

    ``method`` is ``'wood'``, ``'patchy'``, ``'hill'`` or ``'brie'``.  Brie
    needs to know which phase is gas: pass ``gas_index``, or leave it and the
    softest phase is taken as the gas.
    """
    k = np.atleast_1d(np.asarray(moduli, dtype=float))
    s = _normalised(saturations, "saturations")
    key = str(method).strip().lower()

    if key == "brie":
        index = int(np.argmin(k)) if gas_index is None else int(gas_index)
        gas_saturation = float(s[index])
        liquid_mask = np.ones(k.size, dtype=bool)
        liquid_mask[index] = False
        if not liquid_mask.any():
            return float(k[index])
        # The liquids themselves mix by Wood before Brie mixes in the gas.
        k_liquid = wood(k[liquid_mask], s[liquid_mask]) if liquid_mask.sum() > 1 \
            else float(k[liquid_mask][0])
        return float(brie(k_liquid, float(k[index]), 1.0 - gas_saturation,
                          exponent=brie_exponent))

    try:
        return float(FLUID_MIXING_LAWS[key](k, s))
    except KeyError:
        raise ValueError(
            f"unknown fluid mixing law {method!r}; expected one of "
            f"{sorted(list(FLUID_MIXING_LAWS) + ['brie'])}"
        ) from None


def fluid_density(densities, saturations):
    """Effective pore-fluid density: always the volume-weighted average.

    Mass adds however the phases are arranged, so there is no choice of law
    here — only the moduli depend on the geometry.
    """
    rho = np.atleast_1d(np.asarray(densities, dtype=float))
    s = _normalised(saturations, "saturations")
    if rho.shape != s.shape:
        raise ValueError("densities and saturations must have the same shape")
    return float(np.sum(s * rho))


# ------------------------------------------------------------ minerals -----
def _walpole_k(k, g, f, z):
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 / np.sum(f / (k + 4.0 * z / 3.0)) - 4.0 * z / 3.0


def _walpole_zeta(k, g):
    if k + 2.0 * g <= 0:
        return 0.0
    return g / 6.0 * (9.0 * k + 8.0 * g) / (k + 2.0 * g)


def _walpole_g(k, g, f, zeta):
    with np.errstate(divide="ignore", invalid="ignore"):
        if zeta <= 0:
            return 0.0
        return 1.0 / np.sum(f / (g + zeta)) - zeta


def hashin_shtrikman_walpole(moduli_k, moduli_g, fractions):
    """Hashin-Shtrikman-Walpole bounds for a mixture of any number of phases.

    The two-phase Hashin-Shtrikman bounds generalise by evaluating the same
    expressions at the stiffest and softest phase in the mixture, which is
    Walpole's extension.  Returns ``{'K_upper', 'K_lower', 'G_upper',
    'G_lower'}`` as floats.
    """
    k = np.atleast_1d(np.asarray(moduli_k, dtype=float))
    g = np.atleast_1d(np.asarray(moduli_g, dtype=float))
    if k.shape != g.shape:
        raise ValueError("bulk and shear moduli must have the same shape")
    f = _normalised(fractions)
    if f.shape != k.shape:
        raise ValueError("fractions must have one entry per phase")

    if k.size == 1 or np.allclose(k, k[0]) and np.allclose(g, g[0]):
        return {"K_upper": float(k[0]), "K_lower": float(k[0]),
                "G_upper": float(g[0]), "G_lower": float(g[0])}

    g_max, g_min = float(g.max()), float(g.min())
    stiff = int(np.argmax(k))
    soft = int(np.argmin(k))

    return {
        "K_upper": float(_walpole_k(k, g, f, g_max)),
        "K_lower": float(_walpole_k(k, g, f, g_min)),
        "G_upper": float(_walpole_g(k, g, f, _walpole_zeta(float(k[stiff]), g_max))),
        "G_lower": float(_walpole_g(k, g, f, _walpole_zeta(float(k[soft]), g_min))),
    }


#: Ways to collapse the mineral bounds to a single effective medium.
MINERAL_MIXING_LAWS = ["voigt", "reuss", "hill", "hs_upper", "hs_lower", "hs_average"]


def mineral_mix(moduli_k, moduli_g, densities, fractions, method="hill"):
    """Effective mineral moduli and density for a multi-mineral matrix.

    ``method`` selects which of the bounds, or which average of them, to use:
    ``'voigt'``, ``'reuss'``, ``'hill'`` (their average), ``'hs_upper'``,
    ``'hs_lower'`` or ``'hs_average'``.

    Returns ``(K, G, rho)`` in GPa, GPa and g/cc.  Density is always the
    volume-weighted average — only the moduli depend on the choice.
    """
    k = np.atleast_1d(np.asarray(moduli_k, dtype=float))
    g = np.atleast_1d(np.asarray(moduli_g, dtype=float))
    rho = np.atleast_1d(np.asarray(densities, dtype=float))
    f = _normalised(fractions)
    for name, array in (("shear moduli", g), ("densities", rho)):
        if array.shape != k.shape:
            raise ValueError(f"{name} must have one entry per mineral")

    rho_eff = float(np.sum(f * rho))
    key = str(method).strip().lower()

    if key == "voigt":
        return float(np.sum(f * k)), float(np.sum(f * g)), rho_eff
    if key == "reuss":
        with np.errstate(divide="ignore", invalid="ignore"):
            k_r = 1.0 / np.sum(f / k) if np.all(k > 0) else 0.0
            g_r = 1.0 / np.sum(f / g) if np.all(g > 0) else 0.0
        return float(k_r), float(g_r), rho_eff
    if key == "hill":
        upper = mineral_mix(k, g, rho, f, "voigt")
        lower = mineral_mix(k, g, rho, f, "reuss")
        return (0.5 * (upper[0] + lower[0]), 0.5 * (upper[1] + lower[1]), rho_eff)

    bounds = hashin_shtrikman_walpole(k, g, f)
    if key == "hs_upper":
        return bounds["K_upper"], bounds["G_upper"], rho_eff
    if key == "hs_lower":
        return bounds["K_lower"], bounds["G_lower"], rho_eff
    if key == "hs_average":
        return (0.5 * (bounds["K_upper"] + bounds["K_lower"]),
                0.5 * (bounds["G_upper"] + bounds["G_lower"]), rho_eff)

    raise ValueError(
        f"unknown mineral mixing law {method!r}; expected one of {MINERAL_MIXING_LAWS}"
    )
