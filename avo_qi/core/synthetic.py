"""Synthetic angle gathers: reflection-coefficient series to wiggle traces.

The gather is an ``(n_samples, n_angles)`` array in two-way time.  Each
column is the reflection-coefficient series for one incidence angle,
convolved with the wavelet.
"""

from __future__ import annotations

import numpy as np

from .reflectivity import reflectivity_series

__all__ = [
    "convolve_series",
    "build_gather",
    "angle_stack",
    "full_stack",
    "twt_axis",
]


def convolve_series(rc, wavelet):
    """Convolve every column of an RC matrix with ``wavelet`` (mode='same')."""
    rc = np.atleast_2d(np.asarray(rc, dtype=float))
    wavelet = np.asarray(wavelet, dtype=float).ravel()
    if wavelet.size == 0:
        raise ValueError("wavelet is empty")
    out = np.empty_like(rc)
    for j in range(rc.shape[1]):
        out[:, j] = np.convolve(rc[:, j], wavelet, mode="same")
    return out


def build_gather(vp, vs, rho, angles, wavelet, dt=0.001, method="zoeppritz"):
    """Build a synthetic angle gather from Vp/Vs/RHOB logs in TWT.

    Parameters
    ----------
    vp, vs, rho : array_like
        Logs sampled on a regular two-way-time grid (m/s, m/s, g/cc).
    angles : array_like
        Incidence angles in degrees.
    wavelet : array_like
        Wavelet amplitudes, sampled at ``dt``.
    dt : float
        Sample interval in seconds; carried for the time axis only.
    method : str or callable
        ``'zoeppritz'`` or ``'aki_richards'``.

    Returns
    -------
    ndarray
        Gather of shape ``(n_samples, n_angles)``.
    """
    rc = reflectivity_series(vp, vs, rho, angles, method=method)
    return convolve_series(rc, wavelet)


def twt_axis(n_samples, dt, t0=0.0):
    """Regular two-way-time axis in seconds."""
    return t0 + np.arange(int(n_samples)) * float(dt)


def angle_stack(gather, angle_range, angles=None):
    """Mean amplitude over an angle band (near / mid / far stack).

    ``angle_range`` is an inclusive ``(lo, hi)`` pair.  When ``angles`` is
    given the band is selected by angle value in degrees; otherwise the pair
    is read as column indices.
    """
    gather = np.atleast_2d(np.asarray(gather, dtype=float))
    lo, hi = float(angle_range[0]), float(angle_range[1])
    if lo > hi:
        lo, hi = hi, lo

    if angles is None:
        sel = np.zeros(gather.shape[1], dtype=bool)
        sel[int(lo): int(hi) + 1] = True
    else:
        angles = np.asarray(angles, dtype=float)
        if angles.size != gather.shape[1]:
            raise ValueError("angles must have one entry per gather column")
        sel = (angles >= lo) & (angles <= hi)

    if not sel.any():
        raise ValueError(f"no angles fall in the range {angle_range}")
    return gather[:, sel].mean(axis=1)


def full_stack(gather):
    """Mean amplitude over all angles in the gather."""
    gather = np.atleast_2d(np.asarray(gather, dtype=float))
    return gather.mean(axis=1)
