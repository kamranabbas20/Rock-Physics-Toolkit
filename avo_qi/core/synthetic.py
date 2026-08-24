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
    "trace_extrema",
]


def convolve_series(rc, wavelet):
    """Convolve every column of an RC matrix with ``wavelet`` (mode='same')."""
    rc = np.atleast_2d(np.asarray(rc, dtype=float))
    wavelet = np.asarray(wavelet, dtype=float).ravel()
    if wavelet.size == 0:
        raise ValueError("wavelet is empty")
    # np.convolve(mode="same") returns max(len(trace), len(wavelet)) samples,
    # so a window shorter than the wavelet comes back too long. Take the centre
    # of the full convolution instead: identical to "same" for a normal-length
    # trace, and still the trace's length for a short one.
    n = rc.shape[0]
    start = (wavelet.size - 1) // 2
    out = np.empty_like(rc)
    for j in range(rc.shape[1]):
        full = np.convolve(rc[:, j], wavelet, mode="full")
        out[:, j] = full[start:start + n]
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


def _local_extrema(trace):
    """Indices where the trace turns over — its peaks and troughs.

    A turn needs the slope to *reverse*, so only consecutive non-zero slopes
    are compared.  Flat runs are skipped rather than treated as a direction
    change: without that, the shoulder where a flat trace starts descending
    into a spike would be reported as an extremum instead of the spike.
    """
    if trace.size < 3:
        return np.array([], dtype=int)
    slope = np.sign(np.diff(trace))
    # A NaN slope is not a direction, and it must not be read as one: NaN != 0
    # and NaN != NaN are both True, so an unedited gap in the trace turns every
    # sample across it into a "slope reversal" — a dead or partially blank
    # trace would report an extremum at every single sample.
    moving = np.flatnonzero(np.isfinite(slope) & (slope != 0))
    if moving.size < 2:
        return np.array([], dtype=int)
    signs = slope[moving]
    reversals = np.flatnonzero(signs[:-1] != signs[1:]) + 1
    return moving[reversals]


def trace_extrema(trace, samples, half_window=5, polarity=None):
    """Locate the amplitude extremum each reflector produces on a trace.

    A reflection coefficient at interface ``i`` convolved with a zero-phase,
    positive-peak wavelet produces an extremum at sample ``i`` whose sign
    matches the coefficient's.  Interference with neighbouring reflectors
    shifts that extremum, so this searches a window for a genuine turning
    point of the trace and picks the **nearest** one of the **right polarity**
    — not simply the largest amplitude in the window, which would hand a weak
    reflector its loud neighbour's peak and put the marker on the wrong lobe.

    Parameters
    ----------
    trace : array_like
        One trace — an angle stack, a full stack, or a single gather column.
    samples : array_like
        Interface indices, as returned by ``reflector_avo``'s ``sample``.
    half_window : int
        Search radius in samples.  A quarter of the wavelet's dominant period
        is a good default; the caller knows the wavelet, so it is passed in.
    polarity : array_like, optional
        Expected sign per reflector, normally ``sign(R0)``.  Extrema of the
        opposite sign are then rejected.  Omit for a wavelet that is not
        zero-phase positive-peak, where the sign mapping does not hold.

    Returns
    -------
    dict of ndarray
        ``sample`` (the interface), ``index`` (where its extremum sits),
        ``amplitude``, ``polarity`` (+1 peak, -1 trough), ``offset``
        (``index - sample``) and ``is_extremum`` — False when no turning point
        of the right sign was in range, in which case ``index`` falls back to
        the interface itself and the amplitude should not be trusted.
    """
    trace = np.asarray(trace, dtype=float).ravel()
    samples = np.atleast_1d(np.asarray(samples, dtype=int))
    n = trace.size
    half_window = max(int(half_window), 0)

    if n == 0:
        empty_f = np.array([], dtype=float)
        empty_i = np.array([], dtype=int)
        return {"sample": empty_i, "index": empty_i, "amplitude": empty_f,
                "polarity": empty_i, "offset": empty_i,
                "is_extremum": np.array([], dtype=bool)}

    wanted = None
    if polarity is not None:
        sign = np.sign(np.asarray(polarity, dtype=float))
        # A reflector whose R0 could not be computed has an unknown expected
        # sign, not a negative-huge one.  Casting NaN to int is undefined and
        # lands on INT_MIN, which matches no turning point at all — so the
        # reflector was silently reported as having no extremum, on the
        # strength of an integer overflow.  Unknown is the documented
        # no-constraint case, which zero already means.
        wanted = np.where(np.isfinite(sign), sign, 0.0).astype(int)
        if wanted.size != samples.size:
            raise ValueError("polarity must have one entry per sample")

    turning = _local_extrema(trace)
    index = np.empty(samples.size, dtype=int)
    amplitude = np.empty(samples.size, dtype=float)
    is_extremum = np.zeros(samples.size, dtype=bool)

    for k, raw in enumerate(samples):
        s = int(np.clip(raw, 0, n - 1))
        index[k], amplitude[k] = s, trace[s]

        if turning.size == 0:
            continue
        near = turning[np.abs(turning - s) <= half_window]
        if near.size and wanted is not None and wanted[k] != 0:
            near = near[np.sign(trace[near]) == wanted[k]]
        if near.size == 0:
            continue

        # Nearest wins; a tie is broken by the stronger amplitude.
        order = np.lexsort((-np.abs(trace[near]), np.abs(near - s)))
        j = int(near[order[0]])
        index[k], amplitude[k], is_extremum[k] = j, trace[j], True

    # Same guard: a NaN sample on the trace has no polarity to report.
    out_sign = np.sign(amplitude)
    polarity_out = np.where(np.isfinite(out_sign), out_sign, 0.0).astype(int)
    return {
        "sample": samples,
        "index": index,
        "amplitude": amplitude,
        "polarity": polarity_out,
        "offset": index - samples,
        "is_extremum": is_extremum,
    }
