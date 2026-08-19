"""Elastic attributes derived from Vp, Vs and RHOB.

Pure array functions.  Velocities in m/s, densities in g/cc; impedances
therefore come out in (m/s)*(g/cc) and LMR in the square of that, which is
the convention crossplots in the industry are usually drawn in.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "acoustic_impedance",
    "shear_impedance",
    "vpvs",
    "poisson",
    "lambda_rho",
    "mu_rho",
    "lambda_over_mu",
    "eei",
    "attribute_table",
]


def _arr(x):
    return np.asarray(x, dtype=float)


def acoustic_impedance(vp, rho):
    """P-impedance, AI = Vp * rho."""
    return _arr(vp) * _arr(rho)


def shear_impedance(vs, rho):
    """S-impedance, SI = Vs * rho."""
    return _arr(vs) * _arr(rho)


def vpvs(vp, vs):
    """Vp/Vs ratio; zero (or non-finite) Vs yields NaN rather than an inf."""
    vp, vs = _arr(vp), _arr(vs)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = vp / vs
    return np.where(np.isfinite(out), out, np.nan)


def poisson(vp, vs):
    """Poisson's ratio from the velocity ratio."""
    vp, vs = _arr(vp), _arr(vs)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (vp ** 2 - 2 * vs ** 2) / (2 * (vp ** 2 - vs ** 2))
    return np.where(np.isfinite(out), out, np.nan)


def mu_rho(vs, rho):
    """mu*rho = SI^2 (the shear-rigidity axis of an LMR crossplot)."""
    return shear_impedance(vs, rho) ** 2


def lambda_rho(vp, vs, rho):
    """lambda*rho = AI^2 - 2*SI^2 (the incompressibility axis of LMR)."""
    return acoustic_impedance(vp, rho) ** 2 - 2.0 * mu_rho(vs, rho)


def lambda_over_mu(vp, vs):
    """lambda/mu, the fluid-sensitive ratio behind LMR."""
    vp, vs = _arr(vp), _arr(vs)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = vp ** 2 / vs ** 2 - 2.0
    return np.where(np.isfinite(out), out, np.nan)


def eei(vp, vs, rho, chi_deg, vp0=None, vs0=None, rho0=None, k=None):
    """Extended Elastic Impedance (Whitcombe, 2002).

    ``EEI(chi) = vp0 * rho0 * (vp/vp0)^p (vs/vs0)^q (rho/rho0)^r`` with
    ``p = cos(chi) + sin(chi)``, ``q = -8 k sin(chi)`` and
    ``r = cos(chi) - 4 k sin(chi)``.

    The normalising constants default to the log means; ``k`` defaults to the
    mean of ``(vs/vp)^2`` over the log, which is the usual choice.
    """
    vp, vs, rho = _arr(vp), _arr(vs), _arr(rho)

    vp0 = float(np.nanmean(vp)) if vp0 is None else float(vp0)
    vs0 = float(np.nanmean(vs)) if vs0 is None else float(vs0)
    rho0 = float(np.nanmean(rho)) if rho0 is None else float(rho0)

    if k is None:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = (vs / vp) ** 2
        k = float(np.nanmean(np.where(np.isfinite(ratio), ratio, np.nan)))
    k = float(k)

    chi = np.radians(float(chi_deg))
    p = np.cos(chi) + np.sin(chi)
    q = -8.0 * k * np.sin(chi)
    r = np.cos(chi) - 4.0 * k * np.sin(chi)

    with np.errstate(divide="ignore", invalid="ignore"):
        out = vp0 * rho0 * (vp / vp0) ** p * (vs / vs0) ** q * (rho / rho0) ** r
    return np.where(np.isfinite(out), out, np.nan)


def attribute_table(vp, vs, rho, depth=None, chi_deg=None):
    """Assemble the standard QI attribute set as a pandas DataFrame."""
    import pandas as pd

    data = {
        "VP": _arr(vp),
        "VS": _arr(vs),
        "RHOB": _arr(rho),
        "AI": acoustic_impedance(vp, rho),
        "SI": shear_impedance(vs, rho),
        "VPVS": vpvs(vp, vs),
        "POISSON": poisson(vp, vs),
        "LAMBDA_RHO": lambda_rho(vp, vs, rho),
        "MU_RHO": mu_rho(vs, rho),
    }
    if chi_deg is not None:
        data[f"EEI_{chi_deg:g}"] = eei(vp, vs, rho, chi_deg)
    df = pd.DataFrame(data)
    if depth is not None:
        df.insert(0, "DEPTH", _arr(depth))
    return df
