"""Lithology from shale volume, and the lithology pair either side of a reflector.

Lithology is cut from VSH into four classes — sand, silty sand, silt and
shale — at cutoffs the interpreter controls, because the boundaries are a
local calibration rather than a universal constant.

For AVO the useful unit is not a single sample's lithology but the **pair**
across an interface: a "shale over sand" top is a different animal from a
"sand over shale" base, and filtering reflectors by that pair is usually what
an interpreter wants.

Where a well carries no VSH curve, :func:`vsh_from_gr` derives one from gamma
ray against clean-sand and shale reference values.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "LITHOLOGIES",
    "UNDEFINED",
    "DEFAULT_VSH_CUTOFFS",
    "classify_lithology",
    "lithology_fractions",
    "interface_lithology",
    "lobe_lithology",
    "vsh_from_gr",
    "GR_METHODS",
]

#: Lithology classes, ordered cleanest to shaliest.
LITHOLOGIES = ["sand", "silty sand", "silt", "shale"]

#: Label used where VSH is missing, so an absent curve never silently reads
#: as clean sand.
UNDEFINED = "undefined"

#: Upper VSH bound of each class; anything above the last one is shale.
#: These are a starting point, not a standard — recalibrate them per field.
DEFAULT_VSH_CUTOFFS = {"sand": 0.15, "silty sand": 0.35, "silt": 0.60}


def _ordered_cutoffs(cutoffs):
    cutoffs = dict(DEFAULT_VSH_CUTOFFS if cutoffs is None else cutoffs)
    missing = [k for k in LITHOLOGIES[:-1] if k not in cutoffs]
    if missing:
        raise ValueError(f"missing VSH cutoff(s) for: {', '.join(missing)}")
    bounds = [float(cutoffs[k]) for k in LITHOLOGIES[:-1]]
    if any(b <= a for a, b in zip(bounds, bounds[1:])):
        raise ValueError(f"VSH cutoffs must increase; got {bounds}")
    return bounds


def classify_lithology(vsh, cutoffs=None):
    """Label each sample's lithology from its shale volume.

    Parameters
    ----------
    vsh : array_like
        Shale volume, v/v.  Non-finite entries become :data:`UNDEFINED`.
    cutoffs : dict, optional
        Upper VSH bound per class, e.g.
        ``{'sand': 0.15, 'silty sand': 0.35, 'silt': 0.60}``.

    Returns
    -------
    ndarray of object
        One label per sample, drawn from :data:`LITHOLOGIES` or
        :data:`UNDEFINED`.
    """
    bounds = _ordered_cutoffs(cutoffs)
    vsh = np.atleast_1d(np.asarray(vsh, dtype=float))

    labels = np.full(vsh.shape, UNDEFINED, dtype=object)
    good = np.isfinite(vsh)
    if not good.any():
        return labels

    # searchsorted maps each value to the first bound it does not exceed.
    index = np.searchsorted(bounds, vsh[good], side="left")
    labels[good] = np.array(LITHOLOGIES, dtype=object)[np.clip(index, 0, len(LITHOLOGIES) - 1)]
    return labels


def lithology_fractions(labels):
    """Fraction of samples in each lithology, ignoring undefined ones."""
    labels = np.asarray(labels, dtype=object)
    known = labels[labels != UNDEFINED]
    total = known.size
    if total == 0:
        return {name: 0.0 for name in LITHOLOGIES}
    return {name: float(np.sum(known == name)) / total for name in LITHOLOGIES}


def interface_lithology(labels, samples):
    """Lithology either side of each interface, and the pair as one string.

    Interface ``i`` lies between samples ``i`` and ``i + 1``, so the upper
    lithology is ``labels[i]`` and the lower is ``labels[i + 1]``.

    Returns
    -------
    dict of ndarray
        ``upper``, ``lower`` and ``pair`` (``"shale over sand"``).
    """
    labels = np.asarray(labels, dtype=object)
    samples = np.atleast_1d(np.asarray(samples, dtype=int))
    if labels.size == 0:
        empty = np.array([], dtype=object)
        return {"upper": empty, "lower": empty.copy(), "pair": empty.copy()}

    upper_idx = np.clip(samples, 0, labels.size - 1)
    lower_idx = np.clip(samples + 1, 0, labels.size - 1)
    upper = labels[upper_idx]
    lower = labels[lower_idx]
    pair = np.array([f"{a} over {b}" for a, b in zip(upper, lower)], dtype=object)
    return {"upper": upper, "lower": lower, "pair": pair}


def lobe_lithology(labels, bounds, samples=None):
    """Lithology of each event's two half-lobes — the layers AVO was fitted to.

    :func:`interface_lithology` reads the two samples either side of a
    boundary, which is the right answer when the boundary is what was
    measured.  Once the elastic properties come from averaging a half-lobe
    apiece, those two samples are no longer the layers being described: a
    single sample of silt inside eight samples of shale would name the pair
    "silt over sand" when the wave saw shale.  This takes the **commonest**
    label over each half instead, so the pair names the same rock the
    intercept and gradient were computed from.

    ``bounds`` is a :func:`avo_qi.core.blocking.lobe_windows` result.  Where a
    reflector is marked unresolved it has no lobe, and ``samples`` — the
    interface indices — is fallen back to so the pair is still reported.
    """
    labels = np.asarray(labels, dtype=object)
    n = labels.size
    count = np.asarray(bounds["upper_start"]).size
    if n == 0:
        empty = np.array([], dtype=object)
        return {"upper": empty, "lower": empty.copy(), "pair": empty.copy()}

    resolved = np.asarray(bounds.get("resolved", np.ones(count, bool)), dtype=bool)
    samples = None if samples is None else np.asarray(samples, dtype=int)

    def commonest(lo, hi):
        window = labels[max(int(lo), 0):min(int(hi), n)]
        window = window[window != UNDEFINED]
        if window.size == 0:
            return UNDEFINED
        names, counts = np.unique(window.astype(str), return_counts=True)
        return str(names[int(np.argmax(counts))])

    upper = np.empty(count, dtype=object)
    lower = np.empty(count, dtype=object)
    for k in range(count):
        if resolved[k]:
            upper[k] = commonest(bounds["upper_start"][k], bounds["upper_stop"][k])
            lower[k] = commonest(bounds["lower_start"][k], bounds["lower_stop"][k])
        elif samples is not None:
            i = int(np.clip(samples[k], 0, n - 1))
            upper[k] = labels[i]
            lower[k] = labels[min(i + 1, n - 1)]
        else:
            upper[k] = lower[k] = UNDEFINED

    pair = np.array([f"{a} over {b}" for a, b in zip(upper, lower)], dtype=object)
    return {"upper": upper, "lower": lower, "pair": pair}


#: Non-linear gamma-ray to shale-volume transforms, keyed by name.  Each takes
#: the linear gamma-ray index and returns shale volume.
GR_METHODS = {
    "linear": lambda igr: igr,
    "larionov_tertiary": lambda igr: 0.083 * (2.0 ** (3.7 * igr) - 1.0),
    "larionov_older": lambda igr: 0.33 * (2.0 ** (2.0 * igr) - 1.0),
    "steiber": lambda igr: igr / (3.0 - 2.0 * igr),
    "clavier": lambda igr: 1.7 - np.sqrt(3.38 - (igr + 0.7) ** 2),
}


def vsh_from_gr(gr, gr_clean=None, gr_shale=None, method="linear"):
    """Shale volume from a gamma-ray log.

    ``gr_clean`` and ``gr_shale`` default to the 5th and 95th percentiles of
    the log, which is a reasonable first pass but should be set from the
    cleanest sand and the purest shale in the well.

    The non-linear transforms (Larionov, Steiber, Clavier) all read *less*
    shale than the linear index for a given gamma-ray value.
    """
    gr = np.atleast_1d(np.asarray(gr, dtype=float))
    finite = gr[np.isfinite(gr)]
    if finite.size == 0:
        return np.full(gr.shape, np.nan)

    gr_clean = float(np.percentile(finite, 5)) if gr_clean is None else float(gr_clean)
    gr_shale = float(np.percentile(finite, 95)) if gr_shale is None else float(gr_shale)
    if gr_shale == gr_clean:
        raise ValueError("gr_clean and gr_shale must differ")

    try:
        transform = GR_METHODS[str(method).lower()]
    except KeyError:
        raise ValueError(
            f"unknown method {method!r}; expected one of {sorted(GR_METHODS)}"
        ) from None

    igr = np.clip((gr - gr_clean) / (gr_shale - gr_clean), 0.0, 1.0)
    with np.errstate(invalid="ignore"):
        vsh = np.asarray(transform(igr), dtype=float)
    return np.clip(vsh, 0.0, 1.0)
