"""Angle-dependent P-P reflectivity.

Two interface models are provided, both vectorised over an array of
incidence angles given in **degrees** and both returning Rpp for a single
interface described by ``(vp1, vs1, rho1, vp2, vs2, rho2)``:

``zoeppritz_rpp``
    Exact Zoeppritz P-P reflection coefficient in the Aki & Richards
    (2002) convention.
``aki_richards_rpp``
    The three-term linearised approximation.

Unit conventions (see SPEC.md section 3) are the caller's responsibility:
velocities in m/s, densities in g/cc, angles in degrees.  Nothing in this
module converts units.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "zoeppritz_rpp",
    "aki_richards_rpp",
    "reflectivity_series",
    "METHODS",
]

# Guard against exact zeros in shear velocity (a fluid layer): the Zoeppritz
# system divides by vs, and a true fluid is better handled by the limit than
# by a NaN.
_VS_FLOOR = 1e-6


def _snell_angle(p, v):
    """Transmission/conversion angle from the ray parameter, clipped to real.

    ``arcsin`` arguments are clipped to [-1, 1] so that angles past the
    critical angle degrade gracefully (to grazing incidence) instead of
    producing NaNs.
    """
    return np.arcsin(np.clip(p * v, -1.0, 1.0))


def zoeppritz_rpp(vp1, vs1, rho1, vp2, vs2, rho2, theta1):
    """Exact Zoeppritz P-P reflection coefficient.

    Parameters
    ----------
    vp1, vs1, rho1 : float
        Upper (incidence) medium: velocities in m/s, density in g/cc.
    vp2, vs2, rho2 : float
        Lower (transmission) medium.
    theta1 : array_like
        Incidence angles in degrees.

    Returns
    -------
    ndarray
        Rpp, same shape as ``theta1``.

    Notes
    -----
    Ray parameter ``p = sin(theta1) / vp1``; the transmitted P angle and both
    converted S angles follow from Snell's law.  ``arcsin`` arguments are
    clipped to [-1, 1] to guard past the critical angle.
    """
    theta1 = np.radians(np.asarray(theta1, dtype=float))

    vs1 = max(float(vs1), _VS_FLOOR)
    vs2 = max(float(vs2), _VS_FLOOR)
    vp1, vp2 = float(vp1), float(vp2)
    rho1, rho2 = float(rho1), float(rho2)

    p = np.sin(theta1) / vp1          # ray parameter (horizontal slowness)
    theta2 = _snell_angle(p, vp2)     # transmitted P
    phi1 = _snell_angle(p, vs1)       # reflected S
    phi2 = _snell_angle(p, vs2)       # transmitted S

    a = rho2 * (1 - 2 * np.sin(phi2) ** 2) - rho1 * (1 - 2 * np.sin(phi1) ** 2)
    b = rho2 * (1 - 2 * np.sin(phi2) ** 2) + 2 * rho1 * np.sin(phi1) ** 2
    c = rho1 * (1 - 2 * np.sin(phi1) ** 2) + 2 * rho2 * np.sin(phi2) ** 2
    d = 2 * (rho2 * vs2 ** 2 - rho1 * vs1 ** 2)

    E = b * np.cos(theta1) / vp1 + c * np.cos(theta2) / vp2
    F = b * np.cos(phi1) / vs1 + c * np.cos(phi2) / vs2
    G = a - d * np.cos(theta1) / vp1 * np.cos(phi2) / vs2
    H = a - d * np.cos(theta2) / vp2 * np.cos(phi1) / vs1

    D = E * F + G * H * p ** 2

    num = (
        (b * np.cos(theta1) / vp1 - c * np.cos(theta2) / vp2) * F
        - (a + d * np.cos(theta1) / vp1 * np.cos(phi2) / vs2) * H * p ** 2
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        rpp = num / D
    return np.nan_to_num(rpp, nan=0.0, posinf=0.0, neginf=0.0)


def aki_richards_rpp(vp1, vs1, rho1, vp2, vs2, rho2, theta1):
    """Three-term linearised (Aki & Richards) P-P reflection coefficient.

    ``R(t) = 0.5 * (1 - 4 (vs/vp)^2 sin^2 t) * drho/rho
             + 0.5 * dvp/vp * sec^2 t
             - 4 (vs/vp)^2 sin^2 t * dvs/vs``

    with ``vp, vs, rho`` the layer averages and ``d`` the contrast across the
    interface.  Angles in degrees.
    """
    theta1 = np.radians(np.asarray(theta1, dtype=float))

    vp = (float(vp1) + float(vp2)) / 2.0
    vs = (float(vs1) + float(vs2)) / 2.0
    rho = (float(rho1) + float(rho2)) / 2.0

    dvp = float(vp2) - float(vp1)
    dvs = float(vs2) - float(vs1)
    drho = float(rho2) - float(rho1)

    k = (vs / vp) ** 2 if vp != 0 else 0.0
    sin2 = np.sin(theta1) ** 2
    sec2 = 1.0 / np.cos(theta1) ** 2

    term_rho = 0.5 * (1 - 4 * k * sin2) * (drho / rho)
    term_vp = 0.5 * (dvp / vp) * sec2
    term_vs = -4 * k * sin2 * (dvs / vs if vs != 0 else 0.0)

    return term_rho + term_vp + term_vs


METHODS = {
    "zoeppritz": zoeppritz_rpp,
    "aki_richards": aki_richards_rpp,
}


def _resolve_method(method):
    if callable(method):
        return method
    key = str(method).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "zoeppritz": "zoeppritz",
        "exact": "zoeppritz",
        "aki_richards": "aki_richards",
        "aki": "aki_richards",
        "akirichards": "aki_richards",
        "linear": "aki_richards",
    }
    try:
        return METHODS[aliases[key]]
    except KeyError:
        raise ValueError(
            f"unknown reflectivity method {method!r}; "
            f"expected one of {sorted(set(aliases))}"
        ) from None


def reflectivity_series(vp, vs, rho, angles, method="zoeppritz"):
    """Walk a log and build the angle-dependent reflection-coefficient matrix.

    Interface ``i`` sits between sample ``i`` and sample ``i + 1``; the last
    row is zero so the matrix keeps the length of the input logs.

    Returns
    -------
    ndarray
        Shape ``(n_samples, n_angles)``.
    """
    vp = np.asarray(vp, dtype=float)
    vs = np.asarray(vs, dtype=float)
    rho = np.asarray(rho, dtype=float)
    angles = np.atleast_1d(np.asarray(angles, dtype=float))

    if not (vp.shape == vs.shape == rho.shape):
        raise ValueError("vp, vs and rho must have the same shape")
    if vp.ndim != 1:
        raise ValueError("vp, vs and rho must be 1-D logs")

    n = vp.size
    rc = np.zeros((n, angles.size), dtype=float)
    if n < 2:
        return rc

    fn = _resolve_method(method)
    for i in range(n - 1):
        if not np.all(np.isfinite([vp[i], vs[i], rho[i], vp[i + 1], vs[i + 1], rho[i + 1]])):
            continue
        rc[i, :] = fn(vp[i], vs[i], rho[i], vp[i + 1], vs[i + 1], rho[i + 1], angles)
    return rc
