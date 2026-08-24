"""A rock physics model driven sample by sample from VSH, PHIT and SW.

The bounds and frame curves in :mod:`avo_qi.core.rockphysics` are drawn for one
composition over a grid of porosity.  Against a real well that is a weak test:
every sample is compared with the same quartz-and-brine curve, so a shale is
being judged against a sandstone model and the comparison says nothing.

Driving the model from the petrophysical logs fixes that.  Each depth gets its
own matrix from ``VSH``, its own pore fluid from ``SW`` and its own porosity
from ``PHIT``, and the model predicts ``VP``, ``VS`` and ``RHOB`` *logs* to set
against the measured ones::

    VSH  -> mineral mix          -> K_ma, G_ma, rho_ma
    SW   -> fluid mix            -> K_fl, rho_fl
    PHIT -> dry frame            -> K_dry, G_dry
            Gassmann             -> K_sat
            mass balance         -> rho
                                 -> Vp, Vs

**This is not circular.** What would make it circular is deriving the inputs
from the same measurements the output is tested against.  VSH comes from
gamma ray and SW from resistivity; neither knows anything about Vp or Vs, so a
predicted velocity that misses the log is a real disagreement and the model can
still be wrong — which is the only reason to build one.

``PHIT`` is the exception, and :func:`porosity_provenance` exists to catch it.
Density porosity is computed *from* RHOB, so predicting RHOB from it is
guaranteed to succeed and means nothing.  The Vp and Vs comparisons stay valid
either way.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from avo_qi.core.gassmann import gassmann_saturate
from avo_qi.core.rockphysics import (
    FLUIDS,
    MINERALS,
    critical_porosity_dry,
    soft_sand_dry,
    stiff_sand_dry,
    velocities_from_moduli,
)

__all__ = [
    "FRAME_MODELS",
    "MineralLog",
    "FluidLog",
    "PorosityProvenance",
    "mineral_log",
    "fluid_log",
    "forward_model",
    "rock_physics_template",
    "porosity_provenance",
]

#: Dry-frame models the forward model can be built on, keyed by the name the
#: UI shows.  Each takes ``(K_mineral, G_mineral, phi, ...)``.
FRAME_MODELS = {
    "soft sand": soft_sand_dry,
    "stiff sand": stiff_sand_dry,
    "critical porosity": critical_porosity_dry,
}


@dataclass(frozen=True)
class MineralLog:
    """Per-sample matrix properties: moduli in GPa, density in g/cc."""

    K: np.ndarray
    G: np.ndarray
    rho: np.ndarray
    fractions: dict = field(default_factory=dict)


@dataclass(frozen=True)
class FluidLog:
    """Per-sample pore-fluid properties: modulus in GPa, density in g/cc."""

    K: np.ndarray
    rho: np.ndarray


def _mineral_constants(names):
    unknown = [n for n in names if n not in MINERALS]
    if unknown:
        raise ValueError(f"unknown mineral(s) {unknown}; have {sorted(MINERALS)}")
    values = np.array([MINERALS[n] for n in names], dtype=float)
    return values[:, 0], values[:, 1], values[:, 2]


def _vectorised_bounds(fractions, k, g, law):
    """Mineral bounds across every sample at once.

    The mineral *set* is fixed even though the proportions vary, so the
    Hashin-Shtrikman comparison moduli — the stiffest and softest phase — are
    constants and the whole bound collapses to one array expression.  Looping
    per sample would be the obvious way to write this and far too slow to
    redraw a Streamlit page on.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        if law == "voigt":
            return fractions @ k, fractions @ g
        if law == "reuss":
            return 1.0 / (fractions @ (1.0 / k)), 1.0 / (fractions @ (1.0 / g))
        if law == "hill":
            k_v, g_v = _vectorised_bounds(fractions, k, g, "voigt")
            k_r, g_r = _vectorised_bounds(fractions, k, g, "reuss")
            return 0.5 * (k_v + k_r), 0.5 * (g_v + g_r)

        g_hi, g_lo = float(g.max()), float(g.min())
        k_hi, k_lo = float(k[int(np.argmax(k))]), float(k[int(np.argmin(k))])

        def _k_bound(z):
            return 1.0 / (fractions @ (1.0 / (k + 4.0 * z / 3.0))) - 4.0 * z / 3.0

        def _g_bound(k_ref, g_ref):
            if k_ref + 2.0 * g_ref <= 0:
                return np.zeros(fractions.shape[0])
            zeta = g_ref / 6.0 * (9.0 * k_ref + 8.0 * g_ref) / (k_ref + 2.0 * g_ref)
            return 1.0 / (fractions @ (1.0 / (g + zeta))) - zeta

        if law == "hs_upper":
            return _k_bound(g_hi), _g_bound(k_hi, g_hi)
        if law == "hs_lower":
            return _k_bound(g_lo), _g_bound(k_lo, g_lo)
        if law == "hs_average":
            k_u, g_u = _k_bound(g_hi), _g_bound(k_hi, g_hi)
            k_l, g_l = _k_bound(g_lo), _g_bound(k_lo, g_lo)
            return 0.5 * (k_u + k_l), 0.5 * (g_u + g_l)

    raise ValueError(f"unknown mineral mixing law {law!r}")


def mineral_log(vsh, matrix=None, shale="clay", law="hill"):
    """Matrix moduli and density per sample, from a shale-volume log.

    Parameters
    ----------
    vsh : array_like
        Shale volume fraction, clipped to [0, 1].
    matrix : dict, optional
        ``{mineral: fraction}`` for the *non-shale* part of the rock —
        ``{'quartz': 1.0}`` by default.  VSH supplies one degree of freedom,
        so it splits the rock between this blend and ``shale``; anything finer
        has to be fixed here.
    shale : str
        The mineral standing in for the shale fraction.
    law : str
        ``'voigt'``, ``'reuss'``, ``'hill'``, ``'hs_upper'``, ``'hs_lower'`` or
        ``'hs_average'``.

    Returns
    -------
    MineralLog
    """
    v = np.clip(np.atleast_1d(np.asarray(vsh, dtype=float)), 0.0, 1.0)
    matrix = dict(matrix or {"quartz": 1.0})
    if shale in matrix:
        raise ValueError(
            f"{shale!r} is the shale mineral and cannot also be in the matrix blend")
    total = float(sum(matrix.values()))
    if total <= 0:
        raise ValueError("matrix fractions must sum to a positive number")

    names = list(matrix) + [shale]
    k, g, rho = _mineral_constants(names)

    # Columns: the matrix blend scaled by (1 - VSH), then the shale by VSH.
    fractions = np.empty((v.size, len(names)), dtype=float)
    for j, name in enumerate(matrix):
        fractions[:, j] = (matrix[name] / total) * (1.0 - v)
    fractions[:, -1] = v
    # A sample with no VSH at all leaves the composition undefined, not zero.
    fractions[~np.isfinite(np.asarray(vsh, dtype=float)).ravel(), :] = np.nan

    K, G = _vectorised_bounds(fractions, k, g, str(law).strip().lower())
    return MineralLog(K=K, G=G, rho=fractions @ rho,
                      fractions={n: fractions[:, j] for j, n in enumerate(names)})


def fluid_log(sw, brine=None, hydrocarbon="gas", law="wood", brie_exponent=3.0,
              hydrocarbon_properties=None):
    """Pore-fluid modulus and density per sample, from a water-saturation log.

    Parameters
    ----------
    sw : array_like
        Water saturation, clipped to [0, 1].
    brine, hydrocarbon_properties : tuple, optional
        ``(K in GPa, rho in g/cc)``.  Either may be a pair of *arrays* to carry
        properties that vary down the well — Batzle-Wang values along a
        pressure and temperature profile, say.  Default to the fixed table.
    hydrocarbon : str
        Which entry of the fixed table to use when ``hydrocarbon_properties``
        is not given.
    law : str
        ``'wood'`` (fine-scale mixing, the soft limit), ``'patchy'`` (the stiff
        limit), ``'brie'`` or ``'hill'``.

    Returns
    -------
    FluidLog
    """
    s = np.clip(np.atleast_1d(np.asarray(sw, dtype=float)), 0.0, 1.0)
    k_br, rho_br = brine if brine is not None else FLUIDS["brine"]
    if hydrocarbon_properties is not None:
        k_hc, rho_hc = hydrocarbon_properties
    else:
        if hydrocarbon not in FLUIDS:
            raise ValueError(
                f"unknown fluid {hydrocarbon!r}; have {sorted(FLUIDS)}")
        k_hc, rho_hc = FLUIDS[hydrocarbon]
    k_br, rho_br = np.asarray(k_br, float), np.asarray(rho_br, float)
    k_hc, rho_hc = np.asarray(k_hc, float), np.asarray(rho_hc, float)

    key = str(law).strip().lower()
    with np.errstate(divide="ignore", invalid="ignore"):
        if key == "wood":
            k = 1.0 / (s / k_br + (1.0 - s) / k_hc)
        elif key == "patchy":
            k = s * k_br + (1.0 - s) * k_hc
        elif key == "brie":
            k = (k_br - k_hc) * s ** float(brie_exponent) + k_hc
        elif key == "hill":
            k = 0.5 * (1.0 / (s / k_br + (1.0 - s) / k_hc)
                       + s * k_br + (1.0 - s) * k_hc)
        else:
            raise ValueError(
                f"unknown fluid mixing law {law!r}; expected 'wood', 'patchy', "
                "'brie' or 'hill'")

    # Mass adds however the phases are arranged, so density has no choice of law.
    return FluidLog(K=np.broadcast_to(k, s.shape).copy(),
                    rho=np.broadcast_to(s * rho_br + (1.0 - s) * rho_hc,
                                        s.shape).copy())


def forward_model(vsh, phit, sw, matrix=None, shale="clay", mineral_law="hill",
                  frame="soft sand", fluid_law="wood", brie_exponent=3.0,
                  hydrocarbon="gas", brine=None, hydrocarbon_properties=None,
                  phi_c=0.36, n_grains=None, pressure=10e6, shear_factor=1.0):
    """Predict ``VP``, ``VS`` and ``RHOB`` logs from ``VSH``, ``PHIT`` and ``SW``.

    ``pressure`` is the effective pressure in Pa that the granular frame models
    take; it may be an array to follow a depth profile.

    Returns
    -------
    dict
        ``VP``, ``VS``, ``RHOB`` (predicted logs), the intermediates ``K_ma``,
        ``G_ma``, ``RHO_ma``, ``K_fl``, ``RHO_fl``, ``K_dry``, ``G_dry``,
        ``K_sat``, and ``valid``/``reasons`` from the Gassmann step.
    """
    phi = np.atleast_1d(np.asarray(phit, dtype=float))
    minerals = mineral_log(vsh, matrix=matrix, shale=shale, law=mineral_law)
    fluid = fluid_log(sw, brine=brine, hydrocarbon=hydrocarbon, law=fluid_law,
                      brie_exponent=brie_exponent,
                      hydrocarbon_properties=hydrocarbon_properties)

    key = str(frame).strip().lower()
    if key not in FRAME_MODELS:
        raise ValueError(
            f"unknown frame model {frame!r}; expected one of {sorted(FRAME_MODELS)}")
    if key == "critical porosity":
        k_dry, g_dry = critical_porosity_dry(minerals.K, minerals.G, phi, phi_c)
    else:
        k_dry, g_dry = FRAME_MODELS[key](minerals.K, minerals.G, phi, phi_c,
                                         n_grains, pressure, shear_factor)

    saturated = gassmann_saturate(k_dry, minerals.K, fluid.K, phi)
    rho = (1.0 - phi) * minerals.rho + phi * fluid.rho

    # Gassmann leaves the shear modulus alone, so the dry one is the wet one.
    vp, vs = velocities_from_moduli(saturated.values, g_dry, rho)
    vp = np.where(saturated.valid, vp, np.nan)
    vs = np.where(saturated.valid, vs, np.nan)

    return {
        "VP": vp, "VS": vs, "RHOB": np.where(np.isfinite(rho), rho, np.nan),
        "K_ma": minerals.K, "G_ma": minerals.G, "RHO_ma": minerals.rho,
        "K_fl": fluid.K, "RHO_fl": fluid.rho,
        "K_dry": np.asarray(k_dry), "G_dry": np.asarray(g_dry),
        "K_sat": saturated.values,
        "valid": saturated.valid, "reasons": saturated.reasons,
    }


def rock_physics_template(porosities, saturations, vsh=0.0, **kwargs):
    """A rock physics template: AI and Vp/Vs over a porosity–saturation grid.

    The classic QI diagnostic (Ødegaard & Avseth).  Where a bare crossplot
    shows only that some points are softer than others, a template says *how
    porous* and *how wet* a point would have to be to land where it does — it
    turns the axes into something quantitative.

    The grid is built by running :func:`forward_model` itself over every
    combination, rather than by a separate calculation.  That matters: the
    template and the page's per-sample prediction are then the same physics
    with the same mineral law, fluid law, frame model and Gassmann step, so a
    well that misses its own template is telling you something real instead of
    exposing a disagreement between two code paths.

    Parameters
    ----------
    porosities, saturations : array_like
        The grid axes.  Saturation is water saturation, so ``1`` is brine and
        ``0`` is fully hydrocarbon-filled.
    vsh : float
        Shale volume held fixed across the template.  A template is drawn for
        one lithology at a time; a shalier one sits lower and to the right.
    **kwargs
        Passed through to :func:`forward_model` — ``frame``, ``phi_c``,
        ``pressure``, ``hydrocarbon``, ``fluid_law`` and the rest.

    Returns
    -------
    dict
        ``porosity`` and ``saturation`` (the axes), and ``AI``, ``vpvs``,
        ``VP``, ``VS``, ``RHOB`` and ``valid`` each shaped
        ``(n_porosities, n_saturations)``.
    """
    from .attributes import acoustic_impedance, vpvs as _vpvs

    porosities = np.atleast_1d(np.asarray(porosities, dtype=float))
    saturations = np.atleast_1d(np.asarray(saturations, dtype=float))
    shape = (porosities.size, saturations.size)

    phi_grid, sw_grid = np.meshgrid(porosities, saturations, indexing="ij")
    flat_phi = phi_grid.ravel()
    flat_sw = sw_grid.ravel()

    predicted = forward_model(np.full(flat_phi.shape, float(vsh)),
                              flat_phi, flat_sw, **kwargs)

    vp = np.asarray(predicted["VP"], dtype=float).reshape(shape)
    vs = np.asarray(predicted["VS"], dtype=float).reshape(shape)
    rho = np.asarray(predicted["RHOB"], dtype=float).reshape(shape)
    return {
        "porosity": porosities,
        "saturation": saturations,
        "VP": vp, "VS": vs, "RHOB": rho,
        "AI": acoustic_impedance(vp, rho),
        "vpvs": _vpvs(vp, vs),
        "valid": np.asarray(predicted["valid"], dtype=bool).reshape(shape),
    }


# ------------------------------------------------------------ provenance ----
@dataclass(frozen=True)
class PorosityProvenance:
    """Whether a porosity log was computed from the density log.

    ``density_derived`` is ``None`` when the question cannot be answered — most
    often because the porosity barely varies over the selection, which leaves
    nothing to fit a line to.
    """

    density_derived: bool | None
    rho_matrix: float
    rho_fluid: float
    residual: float
    note: str

    @property
    def rhob_check_is_meaningful(self):
        """False only when the density comparison is known to be circular."""
        return self.density_derived is not True


def porosity_provenance(phit, rhob, mnemonic=None, tolerance=0.005,
                        min_spread=0.02):
    """Detect a porosity log that was computed from the density log.

    Density porosity is ``phi = (rho_ma - rho_b) / (rho_ma - rho_fl)``, which
    rearranges to a straight line ``rho_b = rho_ma + phi * (rho_fl - rho_ma)``.
    So fit that line and look at the residual: if a single matrix and fluid
    density reproduce RHOB from PHIT, the porosity carries no information the
    density did not already have, and predicting one from the other proves
    nothing.

    Fitting the two densities rather than assuming them is what makes this
    work on any well — the constants a petrophysicist chose are not known here,
    and guessing them would miss every log that used different ones.

    Parameters
    ----------
    phit, rhob : array_like
        Porosity (v/v) and bulk density (g/cc).
    mnemonic : str, optional
        The source mnemonic the porosity was read from.  Used only to add
        corroboration to ``note``; the numbers decide.
    tolerance : float
        Residual in g/cc below which the fit counts as exact.  Loose enough to
        survive the rounding in a LAS file.
    min_spread : float
        Porosity range below which the question is treated as unanswerable.

    Returns
    -------
    PorosityProvenance
    """
    phi = np.asarray(phit, dtype=float).ravel()
    rho = np.asarray(rhob, dtype=float).ravel()
    if phi.shape != rho.shape:
        raise ValueError("porosity and density must have the same length")

    good = np.isfinite(phi) & np.isfinite(rho)
    nan = float("nan")
    if good.sum() < 3:
        return PorosityProvenance(
            None, nan, nan, nan,
            "Too few samples with both porosity and density to tell whether "
            "the porosity was derived from the density.")

    spread = float(np.ptp(phi[good]))
    if spread < min_spread:
        return PorosityProvenance(
            None, nan, nan, nan,
            f"Porosity varies by only {spread:.3f} v/v over this selection, "
            "which is too little to tell whether it was derived from the "
            "density log. Widen the zone or lithology filter to find out.")

    slope, intercept = np.polyfit(phi[good], rho[good], 1)
    rho_matrix = float(intercept)
    rho_fluid = float(intercept + slope)
    residual = float(np.sqrt(np.mean(
        (rho[good] - (intercept + slope * phi[good])) ** 2)))

    # A tight fit is only evidence if the densities it implies are a real rock
    # and a real fluid; a coincidental straight line is not density porosity.
    plausible = (2.0 <= rho_matrix <= 3.2) and (0.0 <= rho_fluid <= 1.3)
    derived = bool(residual < tolerance and plausible)

    hint = ""
    if mnemonic:
        upper = str(mnemonic).upper()
        if any(tag in upper for tag in ("PHID", "DPHI", "RHOB")):
            hint = f" The source mnemonic {mnemonic!r} points the same way."

    if derived:
        note = (
            f"PHIT reproduces RHOB to {residual:.4f} g/cc from a matrix of "
            f"{rho_matrix:.2f} and a fluid of {rho_fluid:.2f} g/cc, so it is "
            "density porosity. The density comparison below is circular and "
            "says nothing; the Vp and Vs comparisons are unaffected." + hint)
    elif plausible:
        note = (
            f"PHIT does not reproduce RHOB from a single pair of densities "
            f"(residual {residual:.4f} g/cc), so it carries information the "
            "density log does not. All three comparisons are meaningful.")
    else:
        note = (
            "PHIT and RHOB do not lie on a physically sensible density-porosity "
            f"line (it would need a matrix of {rho_matrix:.2f} g/cc), so the "
            "porosity was not derived from the density. All three comparisons "
            "are meaningful.")
    return PorosityProvenance(derived, rho_matrix, rho_fluid, residual, note)
