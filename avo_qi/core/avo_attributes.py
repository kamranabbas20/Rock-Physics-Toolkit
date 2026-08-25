"""AVO attributes: what the intercept and gradient are worth once you have them.

A and B are a coordinate system, not an answer.  Everything here is a way of
reading a point in that plane against something — a background trend, a
lithology trend, or a rock property measured in the well.

Three ideas, in the order they build on each other.

**Pseudo-shear reflectivity.**  A is the P-impedance reflectivity outright;
S-impedance reflectivity is not, but falls out of A and B once a background
Vp/Vs is assumed.  At Vp/Vs = 2 the density terms cancel exactly and
``Rs = (A - B) / 2`` with no further assumption, which is why that ratio is
the one every textbook writes the relation at.

**The fluid factor.**  Smith & Gidlow (1987): the part of the P reflectivity
that the mudrock line does not explain.  A brine-filled clastic sitting on
Castagna's trend has none; a hydrocarbon sand departs from the trend and shows
up.  See :func:`fluid_factor` for what "none" really means — the reflectivity
form leaves a density residual, and pretending otherwise is how a
compaction contrast gets read as a fluid.

**Rotation.**  Any direction in the A-B plane is an attribute:
``A cos(chi) + B sin(chi)``.  Some angles have names — 0 is A, 90 is B, 45 is
the scaled Poisson reflectivity — but the useful ones are found, not named.
:func:`trend_chi` reads them off the background trend, and :func:`chi_sweep`
finds the angle best correlated with a rock property the well actually
measured.  This is the intercept-gradient counterpart of the EEI sweep in
``core/attributes.eei``, and the two answer the same question in different
domains: which direction in elastic space is this fluid or this facies.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "MUDROCK_SLOPE",
    "NAMED_CHI",
    "pseudo_shear_reflectivity",
    "fluid_factor",
    "chi_rotation",
    "trend_chi",
    "chi_sweep",
    "anomaly_ranking",
]

#: Castagna's mudrock line, ``Vp = 1.16 Vs + 1360``.  The 1.16 is the number
#: the fluid factor subtracts with; it is a global average and a field with
#: its own Vp-Vs trend should use that trend's slope instead.
MUDROCK_SLOPE = 1.16

#: Rotations with names, for labelling an axis.  Everything else is found.
NAMED_CHI = {
    0.0: "intercept A (P reflectivity)",
    45.0: "scaled Poisson reflectivity (A+B)/√2",
    90.0: "gradient B",
}


def _arr(x):
    return np.asarray(x, dtype=float)


def pseudo_shear_reflectivity(A, B, vp_vs=2.0):
    """S-impedance reflectivity implied by an intercept and a gradient.

    From the two-term Aki-Richards form with ``k = (Vs/Vp)^2`` and Gardner's
    density-velocity relation ``rho ~ Vp^0.25``::

        Rs = (A (0.8 + 0.8 k) - B) / (8 k)

    At ``vp_vs = 2`` (``k = 0.25``) the density terms cancel identically and
    this reduces to the familiar ``Rs = (A - B) / 2`` — no Gardner, no
    assumption beyond the linearisation itself.  Away from 2 the Gardner
    substitution is doing real work and the answer inherits its error, which
    is why ``vp_vs`` is a parameter and not a constant: pass the well's own
    background ratio.

    Parameters
    ----------
    A, B : array_like
        Intercept and gradient, per reflector.
    vp_vs : float
        Background Vp/Vs the interface sits in.

    Raises
    ------
    ValueError
        If ``vp_vs`` is not positive.  A zero or negative ratio is not a
        degenerate case to be papered over with a NaN; it is a caller bug.
    """
    vp_vs = float(vp_vs)
    if vp_vs <= 0:
        raise ValueError("vp_vs must be positive")
    k = 1.0 / vp_vs ** 2
    return (_arr(A) * (0.8 + 0.8 * k) - _arr(B)) / (8.0 * k)


def fluid_factor(rp, rs, vp_vs=2.0, slope=MUDROCK_SLOPE):
    """Smith & Gidlow's fluid factor: P reflectivity the mudrock line misses.

    ``dF = Rp - slope * (Vs/Vp) * Rs``.

    On the mudrock line ``Vp = slope * Vs + c``, so ``dVp/Vp`` and ``dVs/Vs``
    are locked together and the *velocity* form of this expression is exactly
    zero.  The reflectivity form is not, and the difference matters:

        dF = ½ (drho/rho) (1 - slope * Vs/Vp)

    for an interface on the trend, which at Vp/Vs = 2 is about 0.42 times the
    density contrast.  **A fluid factor is therefore not read against zero.**
    A compaction boundary with a density step and no fluid change has one, and
    a well with a strong density trend has one everywhere.  Read it against
    its own background — :func:`anomaly_ranking` will do that — or against the
    same attribute in a known brine interval.

    Parameters
    ----------
    rp, rs : array_like
        P- and S-impedance reflectivity.  ``rp`` is the intercept A;
        ``rs`` normally comes from :func:`pseudo_shear_reflectivity`.
    vp_vs : float
        Background Vp/Vs, used for the ``Vs/Vp`` factor.
    slope : float
        The mudrock line's slope.  Fit it to the field where you can: 1.16 is
        a global clastic average and a carbonate or a volcaniclastic section
        is not on it.
    """
    vp_vs = float(vp_vs)
    if vp_vs <= 0:
        raise ValueError("vp_vs must be positive")
    return _arr(rp) - float(slope) * (1.0 / vp_vs) * _arr(rs)


def chi_rotation(A, B, chi_deg):
    """Project the A-B plane onto one direction: ``A cos(chi) + B sin(chi)``.

    ``chi`` is one angle, measured from the A axis towards B, in degrees.  The
    projection is unit-norm, so rotations are comparable with each other and
    with A and B themselves.
    """
    chi = np.radians(float(chi_deg))
    return _arr(A) * np.cos(chi) + _arr(B) * np.sin(chi)


def trend_chi(slope):
    """The two rotations a background trend defines.

    A background trend is the direction the ordinary rock in this well runs
    in.  Projecting **along** it gives what every reflector has in common;
    projecting **across** it gives what an anomalous one has that the others
    do not — which is the same quantity
    :meth:`avo_qi.core.avo.BackgroundTrend.deviation` measures, up to the
    trend's own offset.

    Returns
    -------
    dict
        ``along`` and ``across`` in degrees, or NaN where the trend could not
        be fitted.
    """
    slope = float(slope)
    if not np.isfinite(slope):
        return {"along": np.nan, "across": np.nan}
    along = float(np.degrees(np.arctan2(slope, 1.0)))
    return {"along": along, "across": along + 90.0}


def chi_sweep(A, B, target, chi_deg=None, method="spearman"):
    """Find the rotation of the A-B plane best correlated with a property.

    The AVO counterpart of the EEI chi sweep: rather than naming a direction
    in advance, ask the well which direction its porosity — or its shale
    volume, or its saturation — actually points in.

    Correlation is **Spearman by default**: the relationship between a
    reflection attribute and a rock property is monotonic far more often than
    it is linear, and a handful of reflectors is exactly the sample size where
    one outlier decides a Pearson coefficient.

    Parameters
    ----------
    A, B : array_like
        Intercept and gradient per reflector.
    target : array_like
        The property to correlate against, one value per reflector.
    chi_deg : array_like, optional
        Angles to try; every degree from -90 to 90 by default.  The rotation
        is antisymmetric — chi and chi+180 differ only in sign — so half a
        turn covers every direction, and the sign is reported separately.
    method : {'spearman', 'pearson'}
        Rank or linear correlation.

    Returns
    -------
    dict
        ``chi`` and ``correlation`` (arrays over the sweep), ``best_chi``,
        ``best_correlation`` — signed, so the sign says which way the property
        runs — ``n`` reflectors used, and ``reason`` where no sweep was
        possible.
    """
    A, B, target = _arr(A).ravel(), _arr(B).ravel(), _arr(target).ravel()
    if not (A.size == B.size == target.size):
        raise ValueError("A, B and target must be the same length")

    chi = (np.arange(-90.0, 90.0, 1.0) if chi_deg is None
           else np.atleast_1d(_arr(chi_deg)))
    good = np.isfinite(A) & np.isfinite(B) & np.isfinite(target)
    blank = {"chi": chi, "correlation": np.full(chi.size, np.nan),
             "best_chi": np.nan, "best_correlation": np.nan,
             "n": int(good.sum())}
    if good.sum() < 4:
        return {**blank, "reason": "fewer than four reflectors have both"}
    if np.ptp(target[good]) == 0:
        return {**blank, "reason": "the target is constant"}

    from scipy import stats

    correlate = (stats.spearmanr if str(method) == "spearman"
                 else stats.pearsonr)
    a, b, t = A[good], B[good], target[good]
    out = np.full(chi.size, np.nan)
    for i, angle in enumerate(chi):
        rotated = chi_rotation(a, b, float(angle))
        if np.ptp(rotated) == 0:
            continue
        out[i] = float(correlate(rotated, t)[0])

    if not np.isfinite(out).any():
        return {**blank, "reason": "no rotation varied across the reflectors"}
    best = int(np.nanargmax(np.abs(out)))
    return {"chi": chi, "correlation": out, "best_chi": float(chi[best]),
            "best_correlation": float(out[best]), "n": int(good.sum()),
            "reason": None}


def anomaly_ranking(deviation, labels=None, robust=True, groups=None):
    """Rank reflectors by how far they sit from their own background.

    The raw distance from the background trend is not comparable between
    wells, or between two fluid cases of one well: a noisy hole has a wide
    cloud and every event in it looks anomalous.  Dividing by the spread of
    the deviations themselves fixes that — the score is *how unusual is this
    reflector for this well*, which is the question worth asking of a
    background trend at all.

    The spread is a **median absolute deviation about zero**, scaled to a
    standard deviation, unless ``robust`` is off.  Two choices in that:

    *About zero, not about the median of the deviations.*  The trend is
    already the reference — a reflector on it is by definition ordinary — and
    re-centring on the median would move the zero somewhere the trend never
    put it.  It also makes small samples degenerate: with two reflectors the
    median sits exactly between them and both come out equally anomalous.

    *Median, not standard deviation.*  A standard deviation is inflated by the
    very anomalies being looked for, so the strongest events would quietly
    raise the yardstick they are measured with.

    The scale is only as good as the number of events behind it.  A dozen
    reflectors give a soft one; read a z of 3 from such a well as "the
    strongest thing here", not as a probability.

    Parameters
    ----------
    deviation : array_like
        Signed distance from the trend, from
        :meth:`avo_qi.core.avo.BackgroundTrend.deviation`.
    labels : array_like, optional
        Anything identifying each reflector; returned alongside untouched.
    robust : bool
        Use the MAD (default) rather than the root-mean-square deviation.
    groups : array_like, optional
        Score each group in **its own** scatter — normally the well.  Pooling
        several wells and scaling them together lets the noisiest hole set the
        yardstick for all of them, and a quiet well's best event then vanishes
        into everyone else's scatter. The rank is still global, so the answer
        to "what should I look at first" spans the wells even though the
        scores were made separately. Each group also gets its own trend
        implicitly, because the caller measured the deviations against one.

    Returns
    -------
    dict
        ``z`` (signed, in units of the background scatter), ``rank`` (1 is the
        most anomalous, by absolute score), ``order`` (indices, most anomalous
        first), ``scale`` — one number, or ``{group: number}`` when ``groups``
        is given — and ``labels``.
    """
    deviation = _arr(deviation).ravel()
    z = np.full(deviation.shape, np.nan)

    def spread(values):
        if values.size < 2:
            return np.nan
        if robust:
            # MAD about zero — the trend, not the median deviation, is what a
            # reflector is unusual relative to.
            return float(np.median(np.abs(values))) * 1.4826
        return float(np.sqrt(np.mean(values ** 2)))

    if groups is None:
        good = np.isfinite(deviation)
        scale = spread(deviation[good])
        if np.isfinite(scale) and scale > 0:
            z[good] = deviation[good] / scale
    else:
        groups = np.asarray(groups, dtype=object).ravel()
        if groups.size != deviation.size:
            raise ValueError("groups must have one entry per reflector")
        scale = {}
        for name in dict.fromkeys(groups.tolist()):
            here = (groups == name) & np.isfinite(deviation)
            found = spread(deviation[here])
            scale[name] = found
            if np.isfinite(found) and found > 0:
                z[here] = deviation[here] / found

    order = np.argsort(np.where(np.isfinite(z), -np.abs(z), np.inf), kind="stable")
    rank = np.full(deviation.shape, -1, dtype=int)
    rank[order] = np.arange(1, deviation.size + 1)
    rank = np.where(np.isfinite(z), rank, -1)
    return {"z": z, "rank": rank, "order": order, "scale": scale,
            "labels": None if labels is None else np.asarray(labels)}
