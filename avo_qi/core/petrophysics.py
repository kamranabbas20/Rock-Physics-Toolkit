"""Porosity and water saturation from raw logs, for a well that arrives without.

Most wells reaching a QI workflow already carry a petrophysical interpretation
— `PHIF`, `SW`, `VSH` — done by someone with the core data, the pressure data
and the local knowledge to calibrate it.  **That interpretation should be
used.**  Nothing here beats it, and this module exists for the other case: a
well that carries only the raw measurements, where the alternative to a
transparent first-pass interpretation is no interpretation at all.

So the toolkit asks rather than assumes.  Which curves came from the file and
which were computed here is a fact a reader has to be able to recover, because
the difference matters: a density porosity computed with the wrong matrix
density is wrong everywhere downstream, quietly, and in the same direction.

The three transforms are the standard first pass:

* **Density porosity** — a straight lever rule between matrix and fluid
  density.  Its whole accuracy is the matrix density, and a heavily clipped
  result is that assumption telling you it is wrong.
* **Density-neutron porosity** — the two curves respond to gas in opposite
  directions, so their combination both estimates porosity and reveals gas.
* **Archie**, and **Simandoux** where shale conducts enough to matter.
  Archie's clean-sand assumption is the one that fails first in a shaly
  reservoir, and it fails towards optimism: it reads too much water.

Streamlit-free and dependency-light, like the rest of ``core/``.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "porosity_from_density",
    "porosity_from_density_neutron",
    "effective_porosity",
    "sw_archie",
    "sw_simandoux",
    "MATRIX_DENSITY",
    "FLUID_DENSITY",
]

#: Grain densities in g/cc.  Quartz is the default because a QI target usually
#: is a sandstone; a carbonate needs calcite or dolomite or the porosity comes
#: out several units too high.
MATRIX_DENSITY = {
    "sandstone (quartz)": 2.65,
    "limestone (calcite)": 2.71,
    "dolomite": 2.87,
    "shale": 2.70,
}

#: Densities of what fills the *invaded* zone the density tool reads, in g/cc.
FLUID_DENSITY = {
    "fresh mud filtrate": 1.00,
    "salt mud filtrate": 1.10,
    "oil": 0.80,
    "gas": 0.20,
}


def porosity_from_density(rhob, rho_matrix=2.65, rho_fluid=1.0, clip=True):
    """Porosity from the bulk density log.

    .. math:: \\phi = \\frac{\\rho_{ma} - \\rho_b}{\\rho_{ma} - \\rho_{fl}}

    A lever rule between the grain density and whatever fills the pore, and no
    more accurate than those two numbers: at 2.65 against 1.0, being 0.05 g/cc
    wrong in the matrix moves porosity by about 3 porosity units everywhere.

    Returns
    -------
    dict
        ``phi``, and ``clipped`` — the fraction of finite samples that fell
        outside 0–1 before clipping.  A large ``clipped`` is the matrix density
        saying it is wrong, and is worth surfacing rather than hiding.
    """
    rhob = np.asarray(rhob, dtype=float)
    rho_matrix = float(rho_matrix)
    rho_fluid = float(rho_fluid)
    if rho_matrix == rho_fluid:
        raise ValueError("matrix and fluid density must differ")

    phi = (rho_matrix - rhob) / (rho_matrix - rho_fluid)
    finite = np.isfinite(phi)
    outside = int(np.sum(finite & ((phi < 0.0) | (phi > 1.0))))
    fraction = outside / int(finite.sum()) if finite.any() else 0.0
    if clip:
        phi = np.clip(phi, 0.0, 1.0)
    return {"phi": phi, "clipped": fraction}


def porosity_from_density_neutron(rhob, nphi, rho_matrix=2.65, rho_fluid=1.0,
                                  method="rms", clip=True):
    """Porosity from density and neutron together.

    Gas moves the two curves in opposite directions — it lowers bulk density,
    so density porosity reads high, and it has few hydrogen atoms, so neutron
    porosity reads low.  That crossover is the classic gas indicator, and it is
    also why the two are combined rather than averaged blindly:

    * ``"rms"`` — :math:`\\sqrt{(\\phi_D^2 + \\phi_N^2)/2}`, the usual
      gas-bearing combination.  It sits nearer the higher of the two.
    * ``"average"`` — the arithmetic mean, right in a liquid-filled hole.

    ``nphi`` is taken as a fraction; a curve in percent is detected by its
    magnitude and scaled, because a neutron log arrives both ways.

    Returns
    -------
    dict
        ``phi``, ``phi_density``, ``phi_neutron``, ``separation``
        (:math:`\\phi_D - \\phi_N`, positive where gas is indicated) and
        ``clipped``.
    """
    density = porosity_from_density(rhob, rho_matrix, rho_fluid, clip=clip)
    phi_d = density["phi"]

    neutron = np.asarray(nphi, dtype=float)
    finite = neutron[np.isfinite(neutron)]
    # A neutron log arrives as v/v or as percent; 45 p.u. and 0.45 are the same
    # rock, and only the magnitude tells them apart.
    if finite.size and np.nanmedian(np.abs(finite)) > 1.5:
        neutron = neutron / 100.0
    if clip:
        neutron = np.clip(neutron, 0.0, 1.0)

    if str(method).lower() == "average":
        phi = 0.5 * (phi_d + neutron)
    elif str(method).lower() == "rms":
        phi = np.sqrt(np.clip((phi_d ** 2 + neutron ** 2) / 2.0, 0.0, None))
    else:
        raise ValueError(f"unknown method {method!r}; expected 'rms' or 'average'")

    return {"phi": np.clip(phi, 0.0, 1.0) if clip else phi,
            "phi_density": phi_d, "phi_neutron": neutron,
            "separation": phi_d - neutron, "clipped": density["clipped"]}


def effective_porosity(phi_total, vsh, phi_shale=0.10, clip=True):
    """Total porosity less the part held in the shale.

    .. math:: \\phi_e = \\phi_t - V_{sh}\\,\\phi_{sh}

    Clay-bound water is not producible and is not what a fluid substitution
    should be given, so the effective porosity is the one a QI workflow wants.
    """
    phi = np.asarray(phi_total, dtype=float) - (np.asarray(vsh, dtype=float)
                                                * float(phi_shale))
    return np.clip(phi, 0.0, 1.0) if clip else phi


def sw_archie(rt, phi, rw=0.05, a=1.0, m=2.0, n=2.0, clip=True):
    """Water saturation by Archie's equation.

    .. math:: S_w = \\left(\\frac{a\\,R_w}{\\phi^m\\,R_t}\\right)^{1/n}

    Archie assumes the rock conducts only through the water in its pores.  In
    a shaly sand the clay conducts too, so the measured resistivity is lower
    than the water alone would give and Archie reads **too much water** — it
    is optimistic in the wrong direction, hiding pay rather than inventing it.
    :func:`sw_simandoux` is the correction.

    ``a``, ``m`` and ``n`` are the tortuosity factor, the cementation exponent
    and the saturation exponent; they are core-derived per field, and the
    defaults here are only the textbook clean-sand set.
    """
    rt = np.asarray(rt, dtype=float)
    phi = np.asarray(phi, dtype=float)
    rw = np.asarray(rw, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        formation = float(a) * rw / np.power(phi, float(m))
        sw = np.power(formation / rt, 1.0 / float(n))
    sw = np.where(np.isfinite(sw), sw, np.nan)
    return np.clip(sw, 0.0, 1.0) if clip else sw


def sw_simandoux(rt, phi, vsh, rw=0.05, r_shale=2.0, a=1.0, m=2.0, clip=True):
    """Water saturation with the shale's own conductivity taken out.

    The modified Simandoux (Bardon-Pied) form:

    .. math::
        S_w = \\frac{a R_w}{2\\phi^m}\\left[
              \\sqrt{\\left(\\frac{V_{sh}}{R_{sh}}\\right)^2
              + \\frac{4\\phi^m}{a R_w R_t}} - \\frac{V_{sh}}{R_{sh}}\\right]

    which assumes a saturation exponent of 2 and reduces exactly to Archie
    (with ``n = 2``) as ``vsh`` goes to zero — asserted in the tests, because
    a shaly-sand model that does not is not a correction to anything.

    ``r_shale`` is the resistivity of the nearby shale, read off the log.
    """
    rt = np.asarray(rt, dtype=float)
    phi = np.asarray(phi, dtype=float)
    vsh = np.clip(np.asarray(vsh, dtype=float), 0.0, 1.0)
    rw = np.asarray(rw, dtype=float)
    if float(r_shale) <= 0:
        raise ValueError("the shale resistivity must be positive")

    with np.errstate(divide="ignore", invalid="ignore"):
        phi_m = np.power(phi, float(m))
        shale_term = vsh / float(r_shale)
        under = shale_term ** 2 + 4.0 * phi_m / (float(a) * rw * rt)
        sw = (float(a) * rw / (2.0 * phi_m)) * (np.sqrt(under) - shale_term)
    sw = np.where(np.isfinite(sw), sw, np.nan)
    return np.clip(sw, 0.0, 1.0) if clip else sw
