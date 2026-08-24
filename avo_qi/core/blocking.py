"""Blocking logs to the seismic's own resolution.

An interface coefficient taken from two adjacent log samples is the true layer
contrast only when the boundary is a step.  Real boundaries are gradational,
and then the contrast is split across several samples: no single interface
carries it, and the untuned response comes out far too weak.

The fix is to average each side over the scale the seismic actually resolves —
a **half cycle** of the wavelet, which is the same thickness at which a bed
tunes.  Averaging over that window gives the properties the wavelet effectively
sees above and below the boundary, whatever the boundary's shape.

Two averages are offered.  ``'backus'`` is the elastically correct upscaling
for vertical propagation through fine horizontal layering: density averages
arithmetically, while the P- and S-wave moduli average harmonically.
``'mean'`` is the plain arithmetic average of the logs, which is easier to
reason about but is not what a wave does.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "backus_average",
    "arithmetic_average",
    "half_cycle_samples",
    "lobe_windows",
    "block_properties",
    "blocked_reflectivity",
]


def half_cycle_samples(apparent_period, dt):
    """Half a wavelet cycle expressed in samples — the blocking window."""
    dt = float(dt)
    if dt <= 0:
        raise ValueError("dt must be positive")
    return max(int(round(float(apparent_period) / 2.0 / dt)), 1)


def backus_average(vp, vs, rho):
    """Backus average of a window, for vertical propagation.

    Density averages arithmetically; the P-wave modulus ``rho vp^2`` and the
    shear modulus ``rho vs^2`` average harmonically.  Returns
    ``(vp, vs, rho)`` of the equivalent medium.
    """
    vp = np.asarray(vp, dtype=float)
    vs = np.asarray(vs, dtype=float)
    rho = np.asarray(rho, dtype=float)

    good = np.isfinite(vp) & np.isfinite(vs) & np.isfinite(rho) & (rho > 0)
    if not good.any():
        return np.nan, np.nan, np.nan
    vp, vs, rho = vp[good], vs[good], rho[good]

    rho_eff = float(np.mean(rho))
    m = rho * vp ** 2                       # P-wave modulus
    mu = rho * vs ** 2                      # shear modulus

    with np.errstate(divide="ignore", invalid="ignore"):
        m_eff = 1.0 / np.mean(1.0 / m) if np.all(m > 0) else np.nan
        mu_eff = 1.0 / np.mean(1.0 / mu) if np.all(mu > 0) else 0.0

    vp_eff = float(np.sqrt(m_eff / rho_eff)) if np.isfinite(m_eff) else np.nan
    vs_eff = float(np.sqrt(mu_eff / rho_eff)) if np.isfinite(mu_eff) else np.nan
    return vp_eff, vs_eff, rho_eff


def arithmetic_average(vp, vs, rho):
    """Plain arithmetic average of a window — easy to read, not what a wave does."""
    out = []
    for values in (vp, vs, rho):
        values = np.asarray(values, dtype=float)
        good = np.isfinite(values)
        out.append(float(np.mean(values[good])) if good.any() else np.nan)
    return tuple(out)


_AVERAGES = {"backus": backus_average, "mean": arithmetic_average}


def lobe_windows(trace, extrema_index, polarity=None, max_half_width=None):
    """Half-lobe bounds either side of each reflector's amplitude extremum.

    A reflector shows up on the trace as a lobe — a trough or a peak — running
    from one zero crossing to the next.  Its **upper half**, from the crossing
    above down to the extremum, is the part of the trace the layer above
    produced; its **lower half**, from the extremum down to the crossing below,
    belongs to the layer beneath.  Averaging the logs over those two halves
    gives the layers the seismic actually resolves at that reflector.

    That is a different window from a fixed half cycle of the wavelet.  This
    one is measured on the data, so it narrows where beds are thin or where
    interference squeezes the lobe, and opens up where the reflector stands
    alone.  Each half is about a quarter period rather than a half, so the
    contrasts it produces are sharper.

    Parameters
    ----------
    trace : array_like
        The trace the lobes are read from — normally the full stack, so that
        one blocking serves every angle.
    extrema_index : array_like
        Sample index of each reflector's extremum, from
        :func:`avo_qi.core.synthetic.trace_extrema`.
    polarity : array_like, optional
        Expected sign per reflector.  Defaults to the sign of the trace at the
        extremum, which is the same thing whenever the extremum was resolved.
    max_half_width : int, optional
        Give up after this many samples in either direction.  Without a cap a
        reflector sitting on a long one-sided ramp would swallow most of the
        trace.

    Returns
    -------
    dict of ndarray
        ``upper_start``, ``upper_stop``, ``lower_start``, ``lower_stop`` —
        half-open slice bounds into the trace — with ``resolved`` False where
        no crossing was found in range, in which case the bounds are left at
        the extremum and the caller should fall back to a fixed window rather
        than average an empty or arbitrary slice.
    """
    trace = np.asarray(trace, dtype=float).ravel()
    index = np.atleast_1d(np.asarray(extrema_index, dtype=int))
    n = trace.size
    cap = int(max_half_width) if max_half_width else n

    if polarity is None:
        at = np.clip(index, 0, max(n - 1, 0))
        sign = np.sign(trace[at]) if n else np.zeros(index.size)
    else:
        sign = np.sign(np.asarray(polarity, dtype=float))
    sign = np.where(sign == 0, 1.0, sign)

    out = {k: np.zeros(index.size, dtype=int)
           for k in ("upper_start", "upper_stop", "lower_start", "lower_stop")}
    resolved = np.zeros(index.size, dtype=bool)

    for k, centre in enumerate(index):
        centre = int(np.clip(centre, 0, max(n - 1, 0)))
        out["upper_start"][k] = out["upper_stop"][k] = centre
        out["lower_start"][k] = out["lower_stop"][k] = centre
        if n < 3:
            continue
        want = sign[k]
        # There has to be a lobe to halve.  A flat or zero trace would let both
        # walks stop immediately and hand back an empty window that still
        # looked resolved.
        if not np.isfinite(trace[centre]) or trace[centre] * want <= 0:
            continue

        # Walk up until the trace stops matching the lobe's sign.
        top = centre
        while top - 1 >= 0 and centre - (top - 1) <= cap:
            value = trace[top - 1]
            if not np.isfinite(value) or value * want <= 0:
                break
            top -= 1
        found_top = top > 0 and (centre - top) <= cap and (
            not np.isfinite(trace[top - 1]) or trace[top - 1] * want <= 0)

        # ...and down.
        base = centre
        while base + 1 < n and (base + 1) - centre <= cap:
            value = trace[base + 1]
            if not np.isfinite(value) or value * want <= 0:
                break
            base += 1
        found_base = base < n - 1 and (base - centre) <= cap and (
            not np.isfinite(trace[base + 1]) or trace[base + 1] * want <= 0)

        # A lobe only one sample wide cannot be halved: both windows would
        # collapse onto the extremum itself, the two "layers" would be the
        # same sample, and the contrast would come out as exactly zero — a
        # wrong answer dressed as a narrow one.  Such a reflector is reported
        # unresolved so the caller falls back to the fixed window.
        if not (found_top and found_base) or top >= centre or base <= centre:
            continue
        # Half-open slices: [top, centre + 1) above, [centre, base + 1) below,
        # so the extremum itself belongs to both halves rather than to neither.
        out["upper_start"][k], out["upper_stop"][k] = top, centre + 1
        out["lower_start"][k], out["lower_stop"][k] = centre, base + 1
        resolved[k] = True

    out["resolved"] = resolved
    return out


def block_properties(vp, vs, rho, samples, window, method="backus", guard=0,
                     bounds=None):
    """Average the logs a half cycle either side of each interface.

    Parameters
    ----------
    vp, vs, rho : array_like
        Logs on a regular two-way-time grid.
    samples : array_like
        Interface indices; interface ``i`` lies between samples ``i`` and
        ``i + 1``.
    window : int
        Half-cycle length in samples — see :func:`half_cycle_samples`.
    method : {'backus', 'mean'}
        Averaging rule.
    guard : int
        Samples to skip immediately either side of the boundary, so a
        gradational ramp is excluded from both averages rather than pulling
        them together.  Ignored where ``bounds`` supplies the window, since
        those bounds already say exactly which samples belong to each side.
    bounds : dict, optional
        Explicit per-reflector windows from :func:`lobe_windows` —
        ``upper_start``, ``upper_stop``, ``lower_start``, ``lower_stop`` and
        ``resolved``.  Reflectors marked unresolved fall back to the fixed
        ``window``, so a lobe that could not be found costs that reflector its
        measured window rather than costing it an answer.

    Returns
    -------
    dict of ndarray
        ``vp_upper``, ``vs_upper``, ``rho_upper``, ``vp_lower``, ``vs_lower``,
        ``rho_lower`` and the sample counts each average was taken over.
    """
    vp = np.asarray(vp, dtype=float)
    vs = np.asarray(vs, dtype=float)
    rho = np.asarray(rho, dtype=float)
    samples = np.atleast_1d(np.asarray(samples, dtype=int))
    window = max(int(window), 1)
    guard = max(int(guard), 0)
    n = vp.size

    try:
        average = _AVERAGES[str(method).lower()]
    except KeyError:
        raise ValueError(
            f"unknown blocking method {method!r}; expected {sorted(_AVERAGES)}"
        ) from None

    keys = ("vp_upper", "vs_upper", "rho_upper", "vp_lower", "vs_lower", "rho_lower")
    out = {k: np.full(samples.size, np.nan) for k in keys}
    out["n_upper"] = np.zeros(samples.size, dtype=int)
    out["n_lower"] = np.zeros(samples.size, dtype=int)

    use_bounds = bounds is not None
    if use_bounds:
        resolved = np.asarray(bounds.get(
            "resolved", np.ones(samples.size, dtype=bool)), dtype=bool)
    out["from_lobe"] = np.zeros(samples.size, dtype=bool)

    for k, i in enumerate(samples):
        i = int(np.clip(i, 0, n - 2)) if n >= 2 else 0

        if use_bounds and resolved[k]:
            lo_upper = int(np.clip(bounds["upper_start"][k], 0, n))
            hi_upper = int(np.clip(bounds["upper_stop"][k], 0, n))
            lo_lower = int(np.clip(bounds["lower_start"][k], 0, n))
            hi_lower = int(np.clip(bounds["lower_stop"][k], 0, n))
            out["from_lobe"][k] = True
        else:
            hi_upper = max(i + 1 - guard, 0)             # boundary sits at i | i+1
            lo_upper = max(hi_upper - window, 0)
            lo_lower = min(i + 1 + guard, n)
            hi_lower = min(lo_lower + window, n)

        if hi_upper > lo_upper:
            up = average(vp[lo_upper:hi_upper], vs[lo_upper:hi_upper],
                         rho[lo_upper:hi_upper])
            out["vp_upper"][k], out["vs_upper"][k], out["rho_upper"][k] = up
            out["n_upper"][k] = hi_upper - lo_upper
        if hi_lower > lo_lower:
            low = average(vp[lo_lower:hi_lower], vs[lo_lower:hi_lower],
                          rho[lo_lower:hi_lower])
            out["vp_lower"][k], out["vs_lower"][k], out["rho_lower"][k] = low
            out["n_lower"][k] = hi_lower - lo_lower

    return out


def blocked_reflectivity(vp, vs, rho, samples, angles, window, method="backus",
                         guard=0, reflectivity_method="zoeppritz",
                         mask_post_critical=True, bounds=None):
    """Reflectivity of each interface between its **blocked** layers.

    Blocks a half cycle either side of every interface, then computes
    ``R(theta)`` between those two averaged layers rather than between two
    adjacent samples.  On a gradational boundary the adjacent-sample
    coefficient carries only a fraction of the true contrast; this carries all
    of it.

    The critical angle is taken from the blocked velocities too — using the
    adjacent-sample pair would mask the wrong angles.

    Returns
    -------
    dict
        ``rc`` of shape ``(n_reflectors, n_angles)``, ``critical_angle`` per
        reflector, and the blocked properties from :func:`block_properties`.
    """
    from .reflectivity import METHODS, critical_angle, _resolve_method

    angles = np.atleast_1d(np.asarray(angles, dtype=float))
    samples = np.atleast_1d(np.asarray(samples, dtype=int))
    blocked = block_properties(vp, vs, rho, samples, window=window, method=method,
                               guard=guard, bounds=bounds)

    fn = _resolve_method(reflectivity_method)
    exact = fn is METHODS["zoeppritz"]

    rc = np.full((samples.size, angles.size), np.nan)
    theta_c = np.full(samples.size, np.nan)

    for k in range(samples.size):
        upper = (blocked["vp_upper"][k], blocked["vs_upper"][k], blocked["rho_upper"][k])
        lower = (blocked["vp_lower"][k], blocked["vs_lower"][k], blocked["rho_lower"][k])
        if not all(np.isfinite(v) for v in upper + lower):
            continue
        rc[k, :] = fn(*upper, *lower, angles)
        theta_c[k] = critical_angle(upper[0], lower[0])
        if mask_post_critical and exact and np.isfinite(theta_c[k]):
            rc[k, angles >= theta_c[k]] = np.nan

    out = dict(blocked)
    out["rc"] = rc
    out["critical_angle"] = theta_c
    out["samples"] = samples
    return out
