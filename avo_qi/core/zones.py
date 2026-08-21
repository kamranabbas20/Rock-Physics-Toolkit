"""Zonation: turning a discrete LAS curve or a tops list into named intervals.

Wells arrive zoned in two ways.  Either the LAS carries a **discrete curve** —
``ZONE``, ``FORMATION``, ``MARKER`` — holding an integer code per sample, with
the code-to-name mapping living outside the file; or the zonation comes as a
separate **tops list** of names and depths.  Both end up as the same thing: a
table of intervals with a name, a top and a base.

A zone curve is not a log.  Interpolating it would invent codes that do not
exist, so intervals are built by finding where the code changes, and a zone
that reappears deeper is kept as a separate interval rather than merged with
its earlier occurrence.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "ZONE_MNEMONICS",
    "zones_from_curve",
    "zones_from_tops",
    "assign_zones",
    "zone_of_interface",
    "UNZONED",
]

#: Mnemonics that usually hold a discrete zonation.
ZONE_MNEMONICS = [
    "ZONE", "ZONES", "ZONELOG", "FORMATION", "FORM", "FM", "MARKER",
    "MARKERS", "UNIT", "STRAT", "HORIZON", "FACIES", "LITHO",
]

#: Label for samples outside every named zone.
UNZONED = "unzoned"


def _step(depth):
    """Median sample spacing, used to close the deepest interval."""
    if depth.size < 2:
        return 0.0
    step = float(np.median(np.diff(depth)))
    return step if np.isfinite(step) and step > 0 else 0.0


def _clean_codes(values):
    """Discrete codes as an object array, with non-finite entries as None."""
    values = np.asarray(values)
    if values.dtype.kind in "fc":
        out = np.full(values.shape, None, dtype=object)
        finite = np.isfinite(values.astype(float))
        # Codes are discrete; keep them integral where they genuinely are.
        floats = values.astype(float)[finite]
        integral = np.allclose(floats, np.round(floats))
        out[finite] = (np.round(floats).astype(int) if integral else floats).tolist()
        return out
    return np.array([None if v is None or (isinstance(v, float) and not np.isfinite(v))
                     else v for v in values], dtype=object)


def zones_from_curve(depth, codes, names=None, min_samples=1):
    """Build zone intervals from a discrete curve.

    Parameters
    ----------
    depth : array_like
        Depth (or time) per sample, increasing.
    codes : array_like
        The discrete zone curve.  Non-finite samples fall outside any zone.
    names : dict, optional
        ``{code: name}``.  Codes with no entry keep their own value as the
        name, so an unmapped zonation is still usable.
    min_samples : int
        Drop intervals thinner than this, which removes single-sample
        flickers at zone boundaries.

    Returns
    -------
    pandas.DataFrame
        ``zone``, ``code``, ``top``, ``base``, ``n_samples``, ``first``,
        ``last`` — one row per contiguous interval, in depth order.
    """
    import pandas as pd

    depth = np.asarray(depth, dtype=float)
    codes = _clean_codes(codes)
    if depth.size != codes.size:
        raise ValueError("depth and codes must have the same length")
    if depth.size == 0:
        return pd.DataFrame(columns=["zone", "code", "top", "base", "n_samples",
                                     "first", "last"])

    names = dict(names or {})
    rows = []
    start = 0
    for i in range(1, codes.size + 1):
        ended = i == codes.size or codes[i] != codes[start]
        if not ended:
            continue
        code = codes[start]
        n = i - start
        if code is not None and n >= max(int(min_samples), 1):
            rows.append({
                "zone": names.get(code, str(code)),
                "code": code,
                "top": float(depth[start]),
                # The base is the next sample's depth where there is one, so
                # intervals meet rather than leaving a sample-wide gap. The
                # deepest interval runs one step past the last sample: its base
                # is exclusive, so ending it exactly on that sample would leave
                # the bottom of the well unzoned.
                "base": float(depth[i]) if i < depth.size else float(depth[-1]) + _step(depth),
                "n_samples": int(n),
                "first": int(start),
                "last": int(i - 1),
            })
        start = i
    return pd.DataFrame(rows)


def zones_from_tops(tops, base_depth=None):
    """Build intervals from a tops list: names and the depth each one starts.

    ``tops`` is anything with ``name``/``zone`` and ``top``/``depth`` columns,
    or a mapping of ``{name: top_depth}``.  Each zone runs to the next top;
    the deepest runs to ``base_depth`` if given, otherwise it is left open as
    NaN.
    """
    import pandas as pd

    if isinstance(tops, dict):
        frame = pd.DataFrame({"zone": list(tops), "top": list(tops.values())})
    else:
        frame = pd.DataFrame(tops).copy()
        lower = {c.lower(): c for c in frame.columns}
        name_col = next((lower[c] for c in ("zone", "name", "formation", "marker")
                         if c in lower), None)
        top_col = next((lower[c] for c in ("top", "depth", "md", "tvd")
                        if c in lower), None)
        if name_col is None or top_col is None:
            raise ValueError(
                "tops need a name column (zone/name/formation/marker) and a "
                f"depth column (top/depth/md/tvd); found {list(frame.columns)}"
            )
        frame = frame.rename(columns={name_col: "zone", top_col: "top"})

    frame["top"] = pd.to_numeric(frame["top"], errors="coerce")
    frame = frame.dropna(subset=["top"]).sort_values("top").reset_index(drop=True)
    if frame.empty:
        return pd.DataFrame(columns=["zone", "top", "base"])

    base = frame["top"].shift(-1)
    if base_depth is not None:
        base.iloc[-1] = float(base_depth)
    frame["base"] = base
    return frame[["zone", "top", "base"]]


def assign_zones(depth, zones, unzoned=UNZONED):
    """Label every sample with the zone it falls in.

    A sample belongs to the interval with ``top <= depth < base``; an open
    base runs to the end of the well.
    """
    depth = np.asarray(depth, dtype=float)
    labels = np.full(depth.shape, unzoned, dtype=object)
    if zones is None or len(zones) == 0:
        return labels

    for _, row in zones.iterrows():
        top = float(row["top"])
        base = row.get("base", np.nan)
        base = float(base) if base is not None and np.isfinite(base) else np.inf
        labels[(depth >= top) & (depth < base)] = row["zone"]
    return labels


def zone_of_interface(zone_labels, samples):
    """The zone each interface sits in, and whether it is a zone boundary.

    Interface ``i`` lies between samples ``i`` and ``i + 1``.  Where those two
    samples are in different zones the interface *is* the zone boundary, which
    is usually the reflector an interpreter cares most about.
    """
    labels = np.asarray(zone_labels, dtype=object)
    samples = np.atleast_1d(np.asarray(samples, dtype=int))
    if labels.size == 0:
        empty = np.array([], dtype=object)
        return {"zone": empty, "zone_below": empty.copy(),
                "is_zone_boundary": np.array([], dtype=bool)}

    upper = labels[np.clip(samples, 0, labels.size - 1)]
    lower = labels[np.clip(samples + 1, 0, labels.size - 1)]
    return {
        "zone": upper,
        "zone_below": lower,
        "is_zone_boundary": np.array([a != b for a, b in zip(upper, lower)], dtype=bool),
    }
