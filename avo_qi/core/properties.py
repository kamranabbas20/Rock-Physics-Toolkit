"""Rock properties per reflector, and whether the AVO class depends on them.

The reflector table describes what the *seismic* does at each event — the
intercept, the gradient, the class.  This is the other half of the question an
interpreter actually asks: does the class have anything to do with the rock?
Is Class III where the porosity is, does net-to-gross separate II from III, and
is any of it just depth?

Two pieces, in order.

**The properties.**  A reflector's layers are the two half-lobes its intercept
and gradient were fitted from — not the two samples either side of the
boundary, which stopped being the answer once the elastic properties were
averaged over a lobe.  :func:`lobe_property` averages any log over those same
two windows, so porosity "above" means the porosity of the same rock the
gradient came from.  :func:`net_to_gross` is the same average applied to a
0/1 net flag, which is what a net-to-gross *is*.

**The dependence.**  Whether a property separates the classes is a question
with an answer, and eyeballing a box plot is not it.  :func:`class_dependence`
runs a Kruskal-Wallis test — one-way ANOVA on ranks, so no assumption that
porosity is normally distributed within a class — and reports an effect size
beside the p-value, because with thirty events a p-value on its own says very
little.  Read the caveat on that function before quoting it: picked reflectors
are not independent draws.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "NET_CUTOFFS",
    "lobe_property",
    "net_flag",
    "net_to_gross",
    "reservoir_side",
    "pick_side",
    "class_property_summary",
    "class_dependence",
]

#: Default cutoffs for what counts as *net* rock.  ``vsh`` alone gives net
#: sand; adding ``phi`` and ``sw`` gives net pay.  These are a starting point
#: and a field calibration, not a standard — the same warning as
#: :data:`avo_qi.core.lithology.DEFAULT_VSH_CUTOFFS`.
NET_CUTOFFS = {"vsh": 0.35, "phi": 0.08, "sw": 0.70}

_STATISTICS = {"mean": np.nanmean, "median": np.nanmedian}


def _windows(bounds, samples, count, n):
    """Half-open (upper, lower) slices per event, falling back to the two
    samples either side where the event has no lobe of its own."""
    resolved = np.asarray(bounds.get("resolved", np.ones(count, bool)), dtype=bool)
    samples = None if samples is None else np.asarray(samples, dtype=int)
    for k in range(count):
        if resolved[k]:
            yield ((int(bounds["upper_start"][k]), int(bounds["upper_stop"][k])),
                   (int(bounds["lower_start"][k]), int(bounds["lower_stop"][k])))
        elif samples is not None:
            i = int(np.clip(samples[k], 0, n - 1))
            yield (i, i + 1), (min(i + 1, n - 1), min(i + 2, n))
        else:
            yield (0, 0), (0, 0)


def lobe_property(values, bounds, samples=None, statistic="mean"):
    """Average a log over each reflector's two half-lobes.

    The same windows :func:`avo_qi.core.blocking.lobe_windows` produced and
    the elastic properties were blocked over, so a porosity reported here is
    the porosity of the layer the gradient was fitted to.

    Parameters
    ----------
    values : array_like
        One value per sample of the *time* frame the events were picked on.
    bounds : dict
        A :func:`avo_qi.core.blocking.lobe_windows` result.
    samples : array_like, optional
        Interface index per event, used where a reflector is marked unresolved
        and so has no lobe to halve.
    statistic : {'mean', 'median'}
        ``'median'`` is the robust choice where a log carries spikes; the mean
        is what a net-to-gross needs, and is the default so that
        :func:`net_to_gross` and this function agree.

    Returns
    -------
    dict of ndarray
        ``upper``, ``lower``, and ``contrast`` = lower − upper, the signed
        change across the reflector in the direction the wave travels.
        Windows with no finite sample give NaN rather than zero.
    """
    try:
        reduce = _STATISTICS[str(statistic)]
    except KeyError:
        raise ValueError(
            f"unknown statistic {statistic!r}; expected one of {sorted(_STATISTICS)}"
        ) from None

    values = np.asarray(values, dtype=float).ravel()
    n = values.size
    count = int(np.asarray(bounds["upper_start"]).size)
    upper = np.full(count, np.nan)
    lower = np.full(count, np.nan)

    for k, ((ua, ub), (la, lb)) in enumerate(_windows(bounds, samples, count, n)):
        for slot, (start, stop) in ((upper, (ua, ub)), (lower, (la, lb))):
            window = values[max(start, 0):min(stop, n)]
            window = window[np.isfinite(window)]
            if window.size:
                # nanmean of an all-finite window, but warning-free.
                slot[k] = float(reduce(window))

    return {"upper": upper, "lower": lower, "contrast": lower - upper}


def net_flag(vsh, phi=None, sw=None, cutoffs=None):
    """Per sample: 1 where the rock is net, 0 where it is not, NaN unknown.

    ``vsh`` alone is net sand.  Pass ``phi`` and ``sw`` as well for net pay —
    reservoir that is also porous and hydrocarbon-bearing.  A sample missing
    any curve the criterion uses is **unknown rather than non-net**, so a gap
    in the log reduces the interval a net-to-gross is measured over instead of
    quietly pushing it down.
    """
    cuts = dict(NET_CUTOFFS if cutoffs is None else cutoffs)
    vsh = np.asarray(vsh, dtype=float).ravel()
    flag = np.where(vsh <= float(cuts["vsh"]), 1.0, 0.0)
    known = np.isfinite(vsh)

    for values, key, keep in ((phi, "phi", np.greater_equal),
                              (sw, "sw", np.less_equal)):
        if values is None:
            continue
        values = np.asarray(values, dtype=float).ravel()
        if values.size != vsh.size:
            raise ValueError("every net criterion needs one value per sample")
        with np.errstate(invalid="ignore"):
            flag = np.where(keep(values, float(cuts[key])), flag, 0.0)
        known &= np.isfinite(values)

    return np.where(known, flag, np.nan)


def net_to_gross(vsh, bounds, samples=None, phi=None, sw=None, cutoffs=None):
    """Net-to-gross of each reflector's two half-lobes, in v/v.

    A net-to-gross is the mean of a 0/1 net flag over an interval, so this is
    :func:`lobe_property` of :func:`net_flag` — measured over the same rock the
    AVO response came from rather than over a zone, which is what makes it
    comparable with the class beside it.

    Samples where the criterion cannot be evaluated are left out of both the
    net and the gross, so a half-lobe with no VSH at all reports NaN rather
    than a confident zero.
    """
    return lobe_property(net_flag(vsh, phi=phi, sw=sw, cutoffs=cutoffs),
                         bounds, samples=samples, statistic="mean")


def reservoir_side(vsh_upper, vsh_lower):
    """Which half-lobe is the reservoir: the cleaner one.

    A shale-over-sand top carries its reservoir *below*; a sand-over-shale base
    carries it *above*.  Plotting class against "the porosity below" therefore
    mixes reservoir with seal from one reflector to the next, and the trend
    that ought to be there washes out.

    The rule is deliberately blunt — the side with less shale wins — and the
    choice is returned per reflector so it can be read, argued with, and
    overridden.  Where neither side has a VSH the answer is ``'unknown'``.

    Returns
    -------
    ndarray of object
        ``'above'``, ``'below'`` or ``'unknown'`` per reflector.
    """
    upper = np.asarray(vsh_upper, dtype=float)
    lower = np.asarray(vsh_lower, dtype=float)
    side = np.full(upper.shape, "unknown", dtype=object)

    both = np.isfinite(upper) & np.isfinite(lower)
    side[both & (lower <= upper)] = "below"
    side[both & (lower > upper)] = "above"
    side[~both & np.isfinite(upper)] = "above"
    side[~both & np.isfinite(lower)] = "below"
    return side


def pick_side(side, upper, lower):
    """The value on the side named per reflector by :func:`reservoir_side`."""
    side = np.asarray(side, dtype=object)
    upper = np.asarray(upper, dtype=float)
    lower = np.asarray(lower, dtype=float)
    return np.where(side == "above", upper,
                    np.where(side == "below", lower, np.nan))


def class_property_summary(classes, values, order=None):
    """Per class: how many events, and where the property sits.

    Returns a list of dicts — ``avo_class``, ``n``, ``median``, ``p10``,
    ``p90``, ``mean`` — so the caller decides whether it becomes a DataFrame.
    Classes with no finite value are reported with ``n = 0`` rather than
    dropped, because "Class I has no porosity here" is itself the finding.
    """
    classes = np.asarray(classes, dtype=object)
    values = np.asarray(values, dtype=float)
    if classes.size != values.size:
        raise ValueError("classes and values must be the same length")

    names = list(order) if order is not None else sorted(set(classes.tolist()))
    rows = []
    for name in names:
        picked = values[classes == name]
        picked = picked[np.isfinite(picked)]
        if picked.size:
            rows.append({"avo_class": name, "n": int(picked.size),
                         "median": float(np.median(picked)),
                         "p10": float(np.percentile(picked, 10)),
                         "p90": float(np.percentile(picked, 90)),
                         "mean": float(np.mean(picked))})
        else:
            rows.append({"avo_class": name, "n": 0, "median": np.nan,
                         "p10": np.nan, "p90": np.nan, "mean": np.nan})
    return rows


def class_dependence(classes, values, min_per_group=3):
    """Does the property differ between AVO classes?  Kruskal-Wallis.

    A one-way ANOVA on ranks: it asks whether the classes are draws from the
    same distribution, without assuming that distribution is normal — which
    porosity within a class is not, and net-to-gross, bounded at 0 and 1 and
    usually piled up at both ends, is emphatically not.

    Beside the p-value it returns **epsilon-squared**, the fraction of the
    property's rank variance the class label accounts for,
    ``(H - k + 1) / (n - k)``.  That is the number worth reading. A p-value
    answers "could this be nothing?" and an effect size answers "is it worth
    anything?", and on a few dozen reflectors those come apart constantly.

    .. warning::
       Picked reflectors are **not independent samples**.  Neighbouring events
       in one well see overlapping rock, and a single thick sand can produce
       several of them; a well contributes far fewer independent observations
       than it does rows.  The p-value is therefore optimistic — treat it as a
       ranking of which properties separate the classes best, not as a
       significance test you would defend in print.

    Parameters
    ----------
    classes, values : array_like
        One class label and one property value per reflector.
    min_per_group : int
        Classes with fewer than this many finite values are left out of the
        test and named in ``dropped``; two events cannot say anything about a
        distribution, and including them destabilises H.

    Returns
    -------
    dict
        ``h``, ``p``, ``epsilon_squared``, ``n``, ``k``, ``groups``,
        ``dropped`` — and ``reason`` where the test could not be run at all,
        in which case ``h`` and ``p`` are NaN.
    """
    classes = np.asarray(classes, dtype=object)
    values = np.asarray(values, dtype=float)
    if classes.size != values.size:
        raise ValueError("classes and values must be the same length")

    good = np.isfinite(values)
    classes, values = classes[good], values[good]

    groups, kept, dropped = [], [], []
    for name in sorted(set(classes.tolist())):
        picked = values[classes == name]
        (kept if picked.size >= int(min_per_group) else dropped).append(name)
        if picked.size >= int(min_per_group):
            groups.append(picked)

    blank = {"h": np.nan, "p": np.nan, "epsilon_squared": np.nan,
             "n": int(values.size), "k": len(kept), "groups": kept,
             "dropped": dropped}
    if len(groups) < 2:
        return {**blank, "reason": "fewer than two classes have enough events"}
    if all(np.allclose(g, groups[0][0]) for g in groups):
        return {**blank, "reason": "the property is constant"}

    from scipy import stats

    h, p = stats.kruskal(*groups)
    n = int(sum(g.size for g in groups))
    k = len(groups)
    # Epsilon-squared: H is already on the chi-squared scale, so the effect
    # size is what fraction of the available rank variance it accounts for.
    epsilon = (float(h) - k + 1) / (n - k) if n > k else np.nan
    return {"h": float(h), "p": float(p),
            "epsilon_squared": float(np.clip(epsilon, 0.0, 1.0)),
            "n": n, "k": k, "groups": kept, "dropped": dropped, "reason": None}
