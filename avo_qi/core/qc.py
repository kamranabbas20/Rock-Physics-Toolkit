"""Log quality control: nulls, ranges, physical consistency and spikes.

A real LAS arrives with null sentinels, curves in units its header does not
declare, intervals where a tool was off, and spikes from washouts.  Anything
downstream — reflectivity, gathers, classification — treats what it is given
as physics, so the checks belong here, before any of that runs.

The checks are deliberately conservative: they *flag* rather than repair, and
the caller decides what to drop.  The one exception is :func:`despike`, which
repairs only where it is asked to.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "LAS_NULLS",
    "DEFAULT_RANGES",
    "replace_nulls",
    "range_flags",
    "elastic_flags",
    "depth_qc",
    "spike_flags",
    "despike",
    "curve_summary",
    "qc_flags",
    "completeness",
]

#: Null sentinels seen in LAS files.  A header NULL is honoured too, but these
#: turn up in the data even when the header claims something else.
LAS_NULLS = (-999.25, -999.0, -9999.0, -99999.0, -999999.0, 999.25)

#: Plausible ranges per canonical curve, in core units (m/s, g/cc, v/v, API).
#: Values outside are flagged, never silently clipped — a Vp of 300 m/s is a
#: problem to look at, not a number to fix quietly.
DEFAULT_RANGES = {
    "VP": (1400.0, 7000.0),
    "VS": (200.0, 4500.0),
    "RHOB": (1.5, 3.10),
    "GR": (0.0, 300.0),
    "VSH": (0.0, 1.0),
    "PHI": (0.0, 0.50),
    "SW": (0.0, 1.0),
}

#: Below this Vp/Vs, Poisson's ratio turns negative — physically possible in
#: rare rocks, almost always a bad Vs log.
MIN_VPVS = np.sqrt(2.0)


def replace_nulls(values, nulls=LAS_NULLS, extra=None, tol=1e-6):
    """Turn null sentinels into NaN.

    ``extra`` takes the LAS header's own NULL value where one is declared.
    """
    values = np.asarray(values, dtype=float).copy()
    candidates = list(nulls)
    if extra is not None and np.isfinite(extra):
        candidates.append(float(extra))
    for null in candidates:
        values[np.isclose(values, null, atol=tol, rtol=0.0)] = np.nan
    return values


def range_flags(values, lo, hi):
    """Samples outside ``[lo, hi]``.  NaNs are not flagged — they are missing,
    not wrong."""
    values = np.asarray(values, dtype=float)
    with np.errstate(invalid="ignore"):
        return np.isfinite(values) & ((values < lo) | (values > hi))


def elastic_flags(vp, vs, rho):
    """Physical consistency checks across Vp, Vs and density.

    Returns a dict of per-sample boolean masks:

    ``vs_exceeds_vp``
        Shear faster than compressional — impossible; usually swapped curves.
    ``vpvs_below_limit``
        Vp/Vs under sqrt(2), so Poisson's ratio is negative.
    ``poisson_out_of_range``
        Poisson outside (-1, 0.5), the bounds of an isotropic elastic solid.
    ``non_positive``
        A zero or negative velocity or density where a number is present.
    """
    vp = np.asarray(vp, dtype=float)
    vs = np.asarray(vs, dtype=float)
    rho = np.asarray(rho, dtype=float)
    present = np.isfinite(vp) & np.isfinite(vs) & np.isfinite(rho)

    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(vs > 0, vp / vs, np.nan)
        poisson = (vp ** 2 - 2 * vs ** 2) / (2 * (vp ** 2 - vs ** 2))

    return {
        "vs_exceeds_vp": present & (vs >= vp),
        "vpvs_below_limit": present & np.isfinite(ratio) & (ratio < MIN_VPVS),
        "poisson_out_of_range": present & np.isfinite(poisson)
        & ((poisson <= -1.0) | (poisson >= 0.5)),
        "non_positive": (np.isfinite(vp) & (vp <= 0))
        | (np.isfinite(vs) & (vs < 0))
        | (np.isfinite(rho) & (rho <= 0)),
    }


def depth_qc(depth):
    """Check the depth axis: monotonic, evenly sampled, no duplicates."""
    depth = np.asarray(depth, dtype=float)
    finite = depth[np.isfinite(depth)]
    report = {
        "n_samples": int(depth.size),
        "n_missing": int(depth.size - finite.size),
        "monotonic_increasing": bool(finite.size > 1 and np.all(np.diff(finite) > 0)),
        "duplicates": int(finite.size - np.unique(finite).size),
        "start": float(finite[0]) if finite.size else np.nan,
        "stop": float(finite[-1]) if finite.size else np.nan,
        "step": np.nan,
        "step_varies": False,
        "gaps": 0,
    }
    if finite.size > 1:
        steps = np.diff(finite)
        step = float(np.median(steps))
        report["step"] = step
        # A step is "irregular" only beyond rounding noise on the median.
        report["step_varies"] = bool(np.any(np.abs(steps - step) > max(abs(step) * 0.01, 1e-9)))
        report["gaps"] = int(np.sum(steps > 1.5 * abs(step))) if step else 0
    return report


def _rolling_median(values, window):
    from scipy.ndimage import median_filter

    values = np.asarray(values, dtype=float)
    filled = np.where(np.isfinite(values), values, np.nan)
    # median_filter cannot see NaN, so interpolate across gaps first and put
    # them back afterwards.
    index = np.arange(filled.size)
    good = np.isfinite(filled)
    if good.sum() < 2:
        return filled
    interpolated = np.interp(index, index[good], filled[good])
    smoothed = median_filter(interpolated, size=int(window), mode="nearest")
    return np.where(good, smoothed, np.nan)


def spike_flags(values, window=11, threshold=5.0):
    """Samples that depart from a rolling median by more than ``threshold``
    robust standard deviations.

    Uses the median absolute deviation, so a handful of spikes cannot inflate
    the scale they are being measured against.
    """
    values = np.asarray(values, dtype=float)
    if values.size < 3:
        return np.zeros(values.shape, dtype=bool)

    window = max(int(window) | 1, 3)          # odd windows keep it centred
    baseline = _rolling_median(values, window)
    residual = values - baseline
    finite = np.isfinite(residual)
    if finite.sum() < 3:
        return np.zeros(values.shape, dtype=bool)

    mad = float(np.median(np.abs(residual[finite] - np.median(residual[finite]))))
    scale = 1.4826 * mad                       # MAD to standard deviation
    if scale <= 0:
        return np.zeros(values.shape, dtype=bool)
    return finite & (np.abs(residual) > float(threshold) * scale)


def despike(values, window=11, threshold=5.0):
    """Replace flagged spikes with the rolling median.  Returns ``(repaired,
    flags)`` so the caller can see what was changed."""
    values = np.asarray(values, dtype=float)
    flags = spike_flags(values, window=window, threshold=threshold)
    if not flags.any():
        return values.copy(), flags
    repaired = values.copy()
    repaired[flags] = _rolling_median(values, max(int(window) | 1, 3))[flags]
    return repaired, flags


def completeness(values):
    """Fraction of samples that carry a finite value."""
    values = np.asarray(values, dtype=float)
    return float(np.isfinite(values).mean()) if values.size else 0.0


def curve_summary(df, ranges=None, units=None):
    """Per-curve statistics and range-check counts, as a DataFrame."""
    import pandas as pd

    ranges = DEFAULT_RANGES if ranges is None else ranges
    units = units or {}
    rows = []
    for name in df.columns:
        values = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        lo, hi = ranges.get(name, (np.nan, np.nan))
        out_of_range = (int(range_flags(values, lo, hi).sum())
                        if np.isfinite(lo) and np.isfinite(hi) else 0)
        rows.append({
            "curve": name,
            "unit": units.get(name, ""),
            "present": completeness(values),
            "n_missing": int(values.size - finite.size),
            "min": float(finite.min()) if finite.size else np.nan,
            "max": float(finite.max()) if finite.size else np.nan,
            "mean": float(finite.mean()) if finite.size else np.nan,
            "expected_min": lo,
            "expected_max": hi,
            "out_of_range": out_of_range,
        })
    return pd.DataFrame(rows)


def qc_flags(df, ranges=None, spike_window=11, spike_threshold=5.0,
             spike_curves=("VP", "VS", "RHOB")):
    """Per-sample QC flags for a standardised well.

    Returns a DataFrame of boolean columns: one ``<CURVE>_out_of_range`` and
    ``<CURVE>_spike`` per checked curve, the elastic consistency checks, and
    an ``any_flag`` column summarising them.
    """
    import pandas as pd

    ranges = DEFAULT_RANGES if ranges is None else ranges
    flags = pd.DataFrame(index=df.index)

    for name, (lo, hi) in ranges.items():
        if name in df.columns:
            values = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
            flags[f"{name}_out_of_range"] = range_flags(values, lo, hi)

    for name in spike_curves:
        if name in df.columns:
            values = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
            flags[f"{name}_spike"] = spike_flags(values, window=spike_window,
                                                 threshold=spike_threshold)

    if all(c in df.columns for c in ("VP", "VS", "RHOB")):
        elastic = elastic_flags(df["VP"].to_numpy(float), df["VS"].to_numpy(float),
                                df["RHOB"].to_numpy(float))
        for name, mask in elastic.items():
            flags[name] = mask

    flags["any_flag"] = flags.to_numpy(dtype=bool).any(axis=1) if len(flags.columns) \
        else np.zeros(len(df), dtype=bool)
    return flags
