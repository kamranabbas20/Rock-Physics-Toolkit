"""Intercept-gradient analysis and AVO classification.

Reflectors are described by the two-term Shuey (1985) form
``R(theta) = A + B sin^2(theta)``: ``A`` is the intercept (normal-incidence
reflectivity) and ``B`` the gradient.  Classes follow Rutherford & Williams
(1989) as extended by Castagna & Swan (1997).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .reflectivity import reflectivity_series

__all__ = [
    "shuey_fit",
    "aki_richards_fit",
    "classify",
    "classify_array",
    "CLASSES",
    "reflector_avo",
    "background_trend",
    "BackgroundTrend",
]

#: Class labels in display order.
CLASSES = ["I", "IIp", "IIn", "III", "IV", "background/other"]


def _as_2d(r):
    """Return (R as 2-D (n_reflectors, n_angles), was_1d flag)."""
    r = np.asarray(r, dtype=float)
    if r.ndim == 1:
        return r[np.newaxis, :], True
    if r.ndim != 2:
        raise ValueError("R must be 1-D (one reflector) or 2-D (n_reflectors, n_angles)")
    return r, False


def _lstsq(design, r2d):
    """Least-squares solve for every reflector at once, NaN-safe."""
    good = np.all(np.isfinite(r2d), axis=1)
    coef = np.full((r2d.shape[0], design.shape[1]), np.nan)
    if good.any():
        sol, *_ = np.linalg.lstsq(design, r2d[good].T, rcond=None)
        coef[good] = sol.T
    return coef


def shuey_fit(R_of_theta, angles):
    """Least-squares two-term Shuey fit: ``R = A + B sin^2(theta)``.

    Parameters
    ----------
    R_of_theta : array_like
        Reflectivity, either 1-D (one reflector) or 2-D
        ``(n_reflectors, n_angles)``.
    angles : array_like
        Incidence angles in degrees.

    Returns
    -------
    (A, B) : tuple
        Floats for a single reflector, arrays for a stack of them.
    """
    r2d, was_1d = _as_2d(R_of_theta)
    angles = np.asarray(angles, dtype=float)
    if angles.size != r2d.shape[1]:
        raise ValueError("angles must have one entry per reflectivity sample")
    if angles.size < 2:
        raise ValueError("at least two angles are needed for an A/B fit")

    sin2 = np.sin(np.radians(angles)) ** 2
    design = np.column_stack([np.ones_like(sin2), sin2])
    coef = _lstsq(design, r2d)

    A, B = coef[:, 0], coef[:, 1]
    return (float(A[0]), float(B[0])) if was_1d else (A, B)


def aki_richards_fit(R_of_theta, angles, third_term=False):
    """Least-squares fit of the linearised Aki-Richards form.

    ``R = A + B sin^2(theta) [+ C sin^2(theta) tan^2(theta)]``.  With
    ``third_term=False`` this is the same design as Shuey and is kept
    separate so the two fits can be compared on identical footing; with
    ``third_term=True`` the far-angle term is included and ``C`` is returned
    as well.
    """
    r2d, was_1d = _as_2d(R_of_theta)
    angles = np.asarray(angles, dtype=float)
    if angles.size != r2d.shape[1]:
        raise ValueError("angles must have one entry per reflectivity sample")

    th = np.radians(angles)
    sin2 = np.sin(th) ** 2
    cols = [np.ones_like(sin2), sin2]
    if third_term:
        if angles.size < 3:
            raise ValueError("at least three angles are needed for the three-term fit")
        with np.errstate(divide="ignore", invalid="ignore"):
            tan2 = np.tan(th) ** 2
        tan2 = np.where(np.isfinite(tan2), tan2, 0.0)
        cols.append(sin2 * tan2)
    design = np.column_stack(cols)
    coef = _lstsq(design, r2d)

    if third_term:
        A, B, C = coef[:, 0], coef[:, 1], coef[:, 2]
        if was_1d:
            return float(A[0]), float(B[0]), float(C[0])
        return A, B, C

    A, B = coef[:, 0], coef[:, 1]
    return (float(A[0]), float(B[0])) if was_1d else (A, B)


def classify(A, B, a_tol=0.02):
    """AVO class from an intercept-gradient pair.

    ``a_tol`` is the half-width of the near-zero intercept band that
    separates Class II from Classes I and III.  It is a tunable band, not a
    hard sign split, and is exposed in the UI.

    Returns one of ``'I'``, ``'IIp'``, ``'IIn'``, ``'III'``, ``'IV'`` or
    ``'background/other'``.
    """
    if not (np.isfinite(A) and np.isfinite(B)):
        return "background/other"
    A, B, a_tol = float(A), float(B), abs(float(a_tol))

    if B < 0:
        if abs(A) <= a_tol:
            return "IIp" if A >= 0 else "IIn"
        if A > a_tol:
            return "I"
        if A < -a_tol:
            return "III"
    elif B > 0 and A < 0:
        return "IV"
    return "background/other"


def classify_array(A, B, a_tol=0.02):
    """Vectorised :func:`classify` over arrays of intercepts and gradients."""
    A = np.atleast_1d(np.asarray(A, dtype=float))
    B = np.atleast_1d(np.asarray(B, dtype=float))
    return np.array([classify(a, b, a_tol) for a, b in zip(A, B)], dtype=object)


@dataclass
class BackgroundTrend:
    """Robust line ``B = slope * A + intercept`` through the A-B cloud."""

    slope: float
    intercept: float
    n_points: int

    def predict(self, A):
        return self.slope * np.asarray(A, dtype=float) + self.intercept

    def deviation(self, A, B):
        """Signed distance of each (A, B) point from the background line."""
        B = np.asarray(B, dtype=float)
        resid = B - self.predict(A)
        return resid / np.sqrt(1.0 + self.slope ** 2)


def background_trend(A, B):
    """Fit a robust background trend through the intercept-gradient cloud.

    Uses a Theil-Sen estimator so that a handful of fluid anomalies do not
    drag the line that they are meant to be measured against.  Falls back to
    ordinary least squares if SciPy is unavailable.
    """
    A = np.asarray(A, dtype=float).ravel()
    B = np.asarray(B, dtype=float).ravel()
    good = np.isfinite(A) & np.isfinite(B)
    A, B = A[good], B[good]

    if A.size < 2:
        return BackgroundTrend(slope=np.nan, intercept=np.nan, n_points=int(A.size))

    try:
        from scipy.stats import theilslopes

        slope, intercept, *_ = theilslopes(B, A)
    except Exception:  # pragma: no cover - SciPy is a declared dependency
        slope, intercept = np.polyfit(A, B, 1)

    return BackgroundTrend(slope=float(slope), intercept=float(intercept), n_points=int(A.size))


def _detect_reflectors(rc, threshold):
    """Indices of interfaces whose peak |R| across angles exceeds `threshold`."""
    strength = np.nanmax(np.abs(rc), axis=1)
    strength = np.where(np.isfinite(strength), strength, 0.0)
    return np.flatnonzero(strength > threshold)


def reflector_avo(
    rc,
    vp=None,
    vs=None,
    rho=None,
    angles=None,
    method="zoeppritz",
    both=True,
    depth=None,
    twt=None,
    threshold=0.01,
    a_tol=0.02,
    third_term=False,
):
    """Fit and classify every reflector in a well.

    Parameters
    ----------
    rc : ndarray or None
        A precomputed ``(n_samples, n_angles)`` reflection-coefficient
        matrix.  Pass ``None`` to compute it from the logs with ``method``.
    vp, vs, rho : array_like
        Logs, required when ``rc`` is None (and used to label the table).
    angles : array_like
        Incidence angles in degrees.
    method : str
        Reflectivity model used to build ``rc`` when it is not supplied.
    both : bool
        Fit with both Shuey and Aki-Richards and report the difference.
        When False only the Shuey fit is run and the Aki-Richards columns
        are omitted.
    depth, twt : array_like, optional
        Per-sample depth (m) and/or two-way time (s), used to label
        reflectors.  Interface ``i`` is labelled with sample ``i``.
    threshold : float
        Minimum peak ``|R|`` across angles for an interface to count as a
        reflector.  Use 0 to keep every non-trivial interface.
    a_tol : float
        Intercept tolerance band passed to :func:`classify`.
    third_term : bool
        Include the far-angle term in the Aki-Richards fit and report ``C``.

    Returns
    -------
    pandas.DataFrame
        One row per detected reflector: sample index, depth/TWT where known,
        ``A_shuey``, ``B_shuey``, ``A_ar``, ``B_ar``, the Shuey-vs-Aki-R
        deltas, and the assigned ``avo_class``.
    """
    import pandas as pd

    if angles is None:
        raise ValueError("angles are required")
    angles = np.atleast_1d(np.asarray(angles, dtype=float))

    if rc is None:
        if vp is None or vs is None or rho is None:
            raise ValueError("provide either an rc matrix or vp/vs/rho logs")
        rc = reflectivity_series(vp, vs, rho, angles, method=method)
    rc = np.atleast_2d(np.asarray(rc, dtype=float))
    if rc.shape[1] != angles.size:
        raise ValueError("rc must have one column per angle")

    idx = _detect_reflectors(rc, threshold)
    cols = {"sample": idx.astype(int)}

    if depth is not None:
        depth = np.asarray(depth, dtype=float)
        cols["depth"] = depth[idx]
    if twt is not None:
        twt = np.asarray(twt, dtype=float)
        cols["twt"] = twt[idx]

    if idx.size == 0:
        empty = ["A_shuey", "B_shuey", "A_ar", "B_ar", "dA", "dB", "avo_class"]
        return pd.DataFrame({**cols, **{c: [] for c in empty}})

    sub = rc[idx, :]
    A_s, B_s = shuey_fit(sub, angles)
    cols["R0"] = sub[:, int(np.argmin(np.abs(angles)))]
    cols["A_shuey"] = A_s
    cols["B_shuey"] = B_s

    if both:
        ar = aki_richards_fit(sub, angles, third_term=third_term)
        A_a, B_a = ar[0], ar[1]
        cols["A_ar"] = A_a
        cols["B_ar"] = B_a
        if third_term:
            cols["C_ar"] = ar[2]
        cols["dA"] = A_s - A_a
        cols["dB"] = B_s - B_a

    cols["avo_class"] = classify_array(A_s, B_s, a_tol=a_tol)

    if vp is not None and vs is not None and rho is not None:
        vp = np.asarray(vp, dtype=float)
        vs = np.asarray(vs, dtype=float)
        rho = np.asarray(rho, dtype=float)
        cols["vp_upper"] = vp[idx]
        cols["vp_lower"] = vp[np.minimum(idx + 1, vp.size - 1)]
        cols["vpvs_upper"] = vp[idx] / vs[idx]
        cols["vpvs_lower"] = (
            vp[np.minimum(idx + 1, vp.size - 1)] / vs[np.minimum(idx + 1, vs.size - 1)]
        )

    return pd.DataFrame(cols)
