"""Angle-dependent P-P reflectivity.

Two interface models are provided, both vectorised over an array of
incidence angles given in **degrees** and both returning Rpp for a single
interface described by ``(vp1, vs1, rho1, vp2, vs2, rho2)``:

``zoeppritz_rpp``
    Exact Zoeppritz P-P reflection coefficient in the Aki & Richards
    (2002) convention.
``aki_richards_rpp``
    The three-term linearised approximation.
``ruger_vti_rpp`` (in :mod:`avo_qi.core.anisotropy`)
    Rüger's VTI form, reached from here as ``method="ruger"`` with an
    ``anisotropy`` argument carrying epsilon and delta down the log.  It is
    kept in its own module because it needs Thomsen parameters that the other
    two do not, and because nothing supplies those by default.

Unit conventions (see SPEC.md section 3) are the caller's responsibility:
velocities in m/s, densities in g/cc, angles in degrees.  Nothing in this
module converts units.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "critical_angle",
    "zoeppritz_rpp",
    "aki_richards_rpp",
    "reflectivity_series",
    "METHODS",
]

# Guard against exact zeros in shear velocity (a fluid layer): the Zoeppritz
# system divides by vs, and a true fluid is better handled by the limit than
# by a NaN.
_VS_FLOOR = 1e-6


def critical_angle(vp1, vp2):
    """Incidence angle (degrees) at which the transmitted P wave grazes.

    Returns NaN when ``vp2 <= vp1``, where no critical angle exists.  Beyond
    this angle the Zoeppritz solution is complex and the two-term Shuey form
    no longer describes the response, so intercept-gradient fits must stop
    short of it.
    """
    vp1, vp2 = float(vp1), float(vp2)
    if not (np.isfinite(vp1) and np.isfinite(vp2)) or vp2 <= vp1 or vp1 <= 0:
        return float("nan")
    return float(np.degrees(np.arcsin(vp1 / vp2)))


def _snell_angle(p, v):
    """Transmission/conversion angle from the ray parameter, clipped to real.

    ``arcsin`` arguments are clipped to [-1, 1] so that angles past the
    critical angle degrade gracefully (to grazing incidence) instead of
    producing NaNs.
    """
    return np.arcsin(np.clip(p * v, -1.0, 1.0))


def zoeppritz_rpp(vp1, vs1, rho1, vp2, vs2, rho2, theta1, post_critical="clip"):
    """Exact Zoeppritz P-P reflection coefficient.

    Parameters
    ----------
    vp1, vs1, rho1 : float
        Upper (incidence) medium: velocities in m/s, density in g/cc.
    vp2, vs2, rho2 : float
        Lower (transmission) medium.
    theta1 : array_like
        Incidence angles in degrees.
    post_critical : {'clip', 'nan'}
        What to return beyond the critical angle, where the true solution is
        complex.  ``'clip'`` keeps the real-valued continuation (the default,
        so convolved gathers stay finite); ``'nan'`` blanks those angles,
        which is what an intercept-gradient fit wants — the real continuation
        spikes past critical and will drag a gradient the wrong way.

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
    rpp = np.nan_to_num(rpp, nan=0.0, posinf=0.0, neginf=0.0)

    if post_critical == "nan":
        theta_c = critical_angle(vp1, vp2)
        if np.isfinite(theta_c):
            rpp = np.where(np.degrees(theta1) >= theta_c, np.nan, rpp)
    elif post_critical != "clip":
        raise ValueError("post_critical must be 'clip' or 'nan'")
    return rpp


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


def _ruger(*args, **kwargs):
    """Imported late so ``anisotropy`` can import from here without a cycle."""
    from .anisotropy import ruger_vti_rpp

    return ruger_vti_rpp(*args, **kwargs)


METHODS = {
    "zoeppritz": zoeppritz_rpp,
    "aki_richards": aki_richards_rpp,
    "ruger": _ruger,
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
        "ruger": "ruger",
        "rueger": "ruger",
        "vti": "ruger",
        "ruger_vti": "ruger",
        "anisotropic": "ruger",
    }
    try:
        return METHODS[aliases[key]]
    except KeyError:
        raise ValueError(
            f"unknown reflectivity method {method!r}; "
            f"expected one of {sorted(set(aliases))}"
        ) from None


def reflectivity_series(vp, vs, rho, angles, method="zoeppritz",
                        post_critical="clip", anisotropy=None):
    """Walk a log and build the angle-dependent reflection-coefficient matrix.

    Interface ``i`` sits between sample ``i`` and sample ``i + 1``; the last
    row is zero so the matrix keeps the length of the input logs.

    ``post_critical`` is passed through to the exact solution; leave it at
    ``'clip'`` for anything that gets convolved, and use ``'nan'`` when the
    result feeds an intercept-gradient fit.

    ``anisotropy`` carries Thomsen parameters down the log for
    ``method='ruger'``: a mapping with ``epsilon`` and ``delta`` arrays the
    same length as ``vp`` (:func:`avo_qi.core.anisotropy.thomsen_logs_from_vsh`
    builds one from a shale-volume curve).  A sample whose anisotropy is *not
    known* — NaN — is read as isotropic here rather than skipped, because
    blanking the interface would silently drop a reflector from the section;
    the honest reading of "unknown" belongs upstream, where a caller decides
    whether it has the curve at all.  It is ignored by the other two methods,
    which have nowhere to put it.

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
    exact = fn is zoeppritz_rpp
    vti = fn is _ruger
    if vti:
        eps, delta = _anisotropy_logs(anisotropy, n)

    for i in range(n - 1):
        if not np.all(np.isfinite([vp[i], vs[i], rho[i], vp[i + 1], vs[i + 1], rho[i + 1]])):
            continue
        if exact:
            rc[i, :] = fn(vp[i], vs[i], rho[i], vp[i + 1], vs[i + 1], rho[i + 1],
                          angles, post_critical=post_critical)
        elif vti:
            rc[i, :] = fn(vp[i], vs[i], rho[i], vp[i + 1], vs[i + 1], rho[i + 1],
                          angles,
                          upper={"epsilon": eps[i], "delta": delta[i]},
                          lower={"epsilon": eps[i + 1], "delta": delta[i + 1]})
        else:
            rc[i, :] = fn(vp[i], vs[i], rho[i], vp[i + 1], vs[i + 1], rho[i + 1], angles)
    return rc


def _anisotropy_logs(anisotropy, n):
    """Epsilon and delta as two length-``n`` arrays, defaulting to isotropy."""
    if anisotropy is None:
        zeros = np.zeros(n, dtype=float)
        return zeros, zeros
    try:
        pair = [np.asarray(anisotropy[key], dtype=float)
                for key in ("epsilon", "delta")]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "anisotropy must be a mapping with 'epsilon' and 'delta' arrays"
        ) from exc
    out = []
    for values in pair:
        values = np.broadcast_to(np.atleast_1d(values), (n,)).astype(float) \
            if values.size == 1 else values
        if values.shape != (n,):
            raise ValueError(
                f"anisotropy arrays must have one value per sample ({n}), "
                f"got {values.shape}")
        out.append(np.where(np.isfinite(values), values, 0.0))
    return out[0], out[1]
