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
    "tops_from_table",
    "assign_zones",
    "zone_of_interface",
    "zone_of_lobe",
    "zone_statistics",
    "zone_event_summary",
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


#: Column names that hold the depth of a zonation row, in priority order.
_ZONATION_DEPTH = ["TOP", "DEPTH", "DEPT", "MD", "TVD", "TVDSS"]
#: ...and the ones that hold its name or code.
_ZONATION_LABEL = ["ZONE", "NAME", "FORMATION", "MARKER"] + ZONE_MNEMONICS


def _zonation_columns(frame):
    """``(depth_column, label_column)`` for a zonation table, or ``(None, None)``."""
    lookup = {}
    for column in frame.columns:
        key = str(column).strip().upper().replace(" ", "").replace("_", "")
        lookup.setdefault(key, column)

    def pick(candidates, taken=None):
        for candidate in candidates:                     # exact match first
            if candidate in lookup and lookup[candidate] != taken:
                return lookup[candidate]
        for candidate in candidates:                     # then a prefix match
            for key, column in lookup.items():
                if key.startswith(candidate) and column != taken:
                    return column
        return None

    depth_col = pick(_ZONATION_DEPTH)
    label_col = pick(_ZONATION_LABEL, taken=depth_col)
    if label_col is None and len(frame.columns) == 2:
        # Two columns and one of them is the depth: the other must be it.
        label_col = next((c for c in frame.columns if c != depth_col), None)
    return depth_col, label_col


def tops_from_table(frame, names=None, min_samples=1):
    """Formation tops from a table, whichever of the two shapes it arrives in.

    A zonation reaches the toolkit as a **tops list** — a name and the depth it
    starts at, which is what a spreadsheet holds — or as a **discrete curve**:
    a code per sample on a depth axis.  A LAS can only ever be the second,
    because its data section is numeric and has nowhere to put a name; the
    zonation comes as codes, and the codes stand in as their own names unless
    ``names`` maps them.

    Which of the two it is, is decided **on the content, not on the column
    names**: if the label repeats on consecutive rows it is a curve, because a
    tops list never lists the same zone twice in a row — that would be a zone
    interrupted by nothing.  Deciding on names instead looks fine until a zone
    curve arrives with its depth mnemonic spelled ``DEPTH`` or ``MD``, which
    a tops reader happily accepts and turns into several thousand one-sample
    tops.
    """
    import pandas as pd

    frame = pd.DataFrame(frame)
    depth_col, label_col = _zonation_columns(frame)
    if depth_col is None or label_col is None:
        raise ValueError(
            "a zonation needs either a name column (zone/name/formation/"
            "marker) and a depth column (top/depth/md/tvd), or a depth and a "
            f"discrete zone curve; found {list(frame.columns)}")

    ordered = frame[[depth_col, label_col]].copy()
    ordered[depth_col] = pd.to_numeric(ordered[depth_col], errors="coerce")
    ordered = ordered.dropna(subset=[depth_col]).sort_values(depth_col)
    if ordered.empty:
        return pd.DataFrame(columns=["zone", "top", "base"])

    labels = ordered[label_col]
    previous = labels.shift()
    repeated = bool((labels.eq(previous) | (labels.isna() & previous.isna())).any())

    if repeated:
        zones = zones_from_curve(
            ordered[depth_col].to_numpy(float), labels.to_numpy(),
            names=names, min_samples=min_samples)
        return zones[["zone", "top", "base"]] if len(zones) else zones

    named = ordered.rename(columns={depth_col: "top", label_col: "zone"})
    if names:
        named["zone"] = [names.get(v, v) for v in named["zone"]]
    return zones_from_tops(named)


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


def _sample_thickness(depth):
    """Thickness each sample stands for, from the midpoints either side.

    Summing these over a set of samples gives a thickness that is right under
    irregular sampling and does not depend on the samples being contiguous —
    which matters because a zone name can reappear deeper in the well, and its
    first-to-last span would then include the rock in between.
    """
    depth = np.asarray(depth, dtype=float)
    n = depth.size
    if n == 0:
        return np.zeros(0)
    if n == 1:
        return np.zeros(1)
    edges = np.empty(n + 1)
    edges[1:-1] = 0.5 * (depth[:-1] + depth[1:])
    edges[0] = depth[0] - 0.5 * (depth[1] - depth[0])
    edges[-1] = depth[-1] + 0.5 * (depth[-1] - depth[-2])
    return np.abs(np.diff(edges))


def _as_bounds(criterion):
    """``(lo, hi)`` from a pair or a ``{"min": .., "max": ..}`` mapping."""
    if isinstance(criterion, dict):
        lo, hi = criterion.get("min"), criterion.get("max")
    else:
        lo, hi = criterion
    return (None if lo is None else float(lo),
            None if hi is None else float(hi))


def _criteria_masks(criteria, curves, n):
    """``(passes, logged)`` for a set of cutoffs.

    ``logged`` marks the samples where every curve the criteria name is
    actually present.  Keeping it separate is the honest part: a sample with no
    VSH is not net, but neither is it demonstrably non-net, and a net-to-gross
    quoted without saying how much of the zone could be judged is a number with
    a hole in it.
    """
    passes = np.ones(n, dtype=bool)
    logged = np.ones(n, dtype=bool)
    for name, criterion in dict(criteria).items():
        if name not in curves:
            raise ValueError(f"no curve {name!r} for the net/pay criteria; "
                             f"have {sorted(curves)}")
        values = np.asarray(curves[name], dtype=float)
        if values.size != n:
            raise ValueError(f"curve {name!r} has {values.size} samples, "
                             f"expected {n}")
        finite = np.isfinite(values)
        logged &= finite
        lo, hi = _as_bounds(criterion)
        ok = finite.copy()
        if lo is not None:
            ok &= values >= lo
        if hi is not None:
            ok &= values <= hi
        passes &= ok
    return passes, logged


def zone_statistics(zone_labels, depth, curves=None, net=None, pay=None,
                    unzoned=UNZONED, include_unzoned=False):
    """Per-zone thickness, net-to-gross and curve averages.

    The zonation on its own is a filter.  This turns it into an answer: how
    thick each zone is, how much of it passes a reservoir cutoff, and what the
    logs average over it.

    Parameters
    ----------
    zone_labels : array_like
        Zone name per sample, as :func:`assign_zones` returns.
    depth : array_like
        Depth per sample, in the same frame as ``zone_labels``.  **Use the
        depth frame, not time**: a fast layer occupies fewer time samples per
        metre, so a net-to-gross counted in time is biased by velocity.
    curves : mapping or DataFrame, optional
        Curves to average, e.g. ``{"VSH": .., "PHI": .., "SW": ..}``.  Averages
        are thickness-weighted and ignore missing samples.
    net, pay : mapping, optional
        Cutoffs as ``{curve: (lo, hi)}`` or ``{curve: {"min": .., "max": ..}}``
        with either bound optional — ``{"VSH": {"max": 0.35}}`` is reservoir,
        adding ``{"SW": {"max": 0.5}}`` is pay.  A sample must pass every named
        cutoff.
    include_unzoned : bool
        Report the samples outside every named zone as their own row.

    Returns
    -------
    pandas.DataFrame
        ``zone``, ``top``, ``base``, ``gross``, then ``net``/``ntg`` and
        ``pay``/``ptg`` where cutoffs were given, the thickness-weighted mean
        of each curve as ``mean_<name>`` (and ``net_mean_<name>`` over the net
        interval), ``n_samples``, and ``net_coverage`` / ``pay_coverage`` — the
        fraction of the zone where those cutoffs could be evaluated at all.
        Coverage is what stops a zone with no Sw curve from reading as zero pay
        rather than as unknown.
    """
    import pandas as pd

    labels = np.asarray(zone_labels, dtype=object)
    depth = np.asarray(depth, dtype=float)
    if labels.size != depth.size:
        raise ValueError("zone_labels and depth must have the same length")

    curves = {} if curves is None else (
        {c: curves[c].to_numpy(dtype=float) for c in curves.columns}
        if hasattr(curves, "columns") else
        {k: np.asarray(v, dtype=float) for k, v in dict(curves).items()})

    columns = (["zone", "top", "base", "gross"]
               + (["net", "ntg"] if net else [])
               + (["pay", "ptg"] if pay else [])
               + [f"mean_{c}" for c in curves]
               + ([f"net_mean_{c}" for c in curves] if net else [])
               + ["n_samples"] + (["net_coverage"] if net else [])
               + (["pay_coverage"] if pay else []))
    if labels.size == 0:
        return pd.DataFrame(columns=columns)

    thickness = _sample_thickness(depth)
    net_pass, net_logged = (_criteria_masks(net, curves, labels.size) if net
                            else (None, None))
    pay_pass, pay_logged = (_criteria_masks(pay, curves, labels.size) if pay
                            else (None, None))

    def average(values, where):
        """Thickness-weighted mean over the samples where the curve exists."""
        use = where & np.isfinite(values)
        if not use.any():
            return float("nan")
        if thickness[use].sum() <= 0:        # a single sample has no thickness
            return float(values[use].mean())
        return float(np.average(values[use], weights=thickness[use]))

    names = [n for n in dict.fromkeys(labels.astype(str))
             if include_unzoned or n != str(unzoned)]
    rows = []
    for name in names:
        here = labels.astype(str) == name
        gross = float(thickness[here].sum())
        row = {"zone": name,
               "top": float(depth[here].min()),
               "base": float(depth[here].max()),
               "gross": gross}
        if net is not None:
            in_net = here & net_pass
            row["net"] = float(thickness[in_net].sum())
            row["ntg"] = row["net"] / gross if gross > 0 else float("nan")
        if pay is not None:
            row["pay"] = float(thickness[here & pay_pass].sum())
            row["ptg"] = row["pay"] / gross if gross > 0 else float("nan")
        for curve, values in curves.items():
            row[f"mean_{curve}"] = average(values, here)
        if net is not None:
            for curve, values in curves.items():
                row[f"net_mean_{curve}"] = average(values, here & net_pass)
        row["n_samples"] = int(here.sum())
        for column, logged in (("net_coverage", net_logged),
                               ("pay_coverage", pay_logged)):
            if logged is not None:
                covered = float(thickness[here & logged].sum())
                row[column] = covered / gross if gross > 0 else float("nan")
        rows.append(row)

    frame = pd.DataFrame(rows, columns=columns)
    return frame.sort_values("top").reset_index(drop=True)


def zone_event_summary(events, zone_columns="zone", class_column="avo_class",
                       deviation_column=None, depth_column=None):
    """What the picked reflectors say about each zone.

    The log side of a zone summary describes the rock; this is the seismic
    side — how many events fall in the zone, what AVO classes they are, and
    which one departs furthest below the background trend.  ``deviation_column``
    is signed and negative below the trend, so the *minimum* is the strongest
    candidate anomaly.

    ``zone_columns`` may name **both sides** of the event, ``("zone",
    "zone_below")``, and usually should.  A reservoir top has its upper lobe in
    the seal, so counting only the upper side files the most interesting event
    a reservoir has under the shale above it — and leaves the reservoir looking
    like it has nothing but weak internal reflections.  An event counted on
    either side appears in both zones, which is the honest reading of a
    boundary: it belongs to the pair.
    """
    import pandas as pd

    columns = ([zone_columns] if isinstance(zone_columns, str)
               else [c for c in zone_columns])
    if events is None or not len(events):
        columns = []
    else:
        columns = [c for c in columns if c in events]
    if not columns:
        return pd.DataFrame(columns=["zone", "n_events", "top_class", "class_mix"])

    sides = [events[c].astype(str) for c in columns]
    order = dict.fromkeys(pd.concat(sides, ignore_index=True))

    rows = []
    for name in order:
        here = sides[0] == name
        for side in sides[1:]:
            here = here | (side == name)
        inside = events[here.to_numpy()]
        row = {"zone": name, "n_events": int(len(inside))}
        if class_column in inside:
            counts = inside[class_column].astype(str).value_counts()
            row["top_class"] = str(counts.index[0])
            row["class_mix"] = ", ".join(f"{k}×{v}" for k, v in counts.items())
        else:
            row["top_class"] = ""
            row["class_mix"] = ""
        if deviation_column and deviation_column in inside:
            values = pd.to_numeric(inside[deviation_column], errors="coerce")
            if values.notna().any():
                worst = values.idxmin()
                row["min_deviation"] = float(values.loc[worst])
                if depth_column and depth_column in inside:
                    row["deviation_depth"] = float(inside.loc[worst, depth_column])
            else:
                row["min_deviation"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def zone_of_lobe(zone_labels, bounds, samples=None, unzoned=UNZONED):
    """The zone either side of an event, read over its two half-lobes.

    :func:`zone_of_interface` compares the two samples straddling a boundary,
    which is right when every interface is a candidate reflector.  Once events
    are picked off the trace there are far fewer of them, and a formation top
    almost never falls exactly between one event's two samples — so that flag
    goes quiet and stops meaning anything.

    An event's lobe is its zone of influence: the samples it was blocked over
    and the ones its amplitude actually came from.  A top falling anywhere
    inside that lobe is a top this event is carrying, which is the question
    worth asking of a reflector.  The upper and lower halves take their
    commonest label, matching :func:`avo_qi.core.lithology.lobe_lithology`.
    """
    labels = np.asarray(zone_labels, dtype=object)
    n = labels.size
    count = np.asarray(bounds["upper_start"]).size
    if n == 0:
        empty = np.array([], dtype=object)
        return {"zone": empty, "zone_below": empty.copy(),
                "is_zone_boundary": np.array([], dtype=bool)}

    resolved = np.asarray(bounds.get("resolved", np.ones(count, bool)), dtype=bool)
    samples = None if samples is None else np.asarray(samples, dtype=int)

    def commonest(lo, hi):
        window = labels[max(int(lo), 0):min(int(hi), n)]
        if window.size == 0:
            return unzoned
        names, counts = np.unique(window.astype(str), return_counts=True)
        return str(names[int(np.argmax(counts))])

    above = np.empty(count, dtype=object)
    below = np.empty(count, dtype=object)
    for k in range(count):
        if resolved[k]:
            above[k] = commonest(bounds["upper_start"][k], bounds["upper_stop"][k])
            below[k] = commonest(bounds["lower_start"][k], bounds["lower_stop"][k])
        elif samples is not None:
            i = int(np.clip(samples[k], 0, n - 1))
            above[k] = labels[i]
            below[k] = labels[min(i + 1, n - 1)]
        else:
            above[k] = below[k] = unzoned

    return {"zone": above, "zone_below": below,
            "is_zone_boundary": above != below}


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
