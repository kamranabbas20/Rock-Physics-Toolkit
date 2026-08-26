"""Shear velocity where a well has none.

A quarter of the wells anyone wants to run through this toolkit have no shear
sonic, and without Vs there is no gradient, no Vp/Vs, no Poisson, no LMR and no
Gassmann — the whole page stops.  Predicting it is standard practice, and it is
also the single largest modelling assumption anyone using this tool will make,
so everything here is built to keep that visible: the prediction is a curve
*beside* the measurement, never on top of it, and it comes back with a mask
saying which samples it invented.

Three ways to get one, in ascending order of how much they know about the
well:

**A published transform.**  :data:`GREENBERG_CASTAGNA` is the industry default:
a Vp-Vs polynomial per lithology, Hill-averaged over the mixture, iterated for
the pore fluid.  Its shape — lithology-keyed polynomial coefficients in km/s —
is general, so any other published set drops into the same function as data
rather than as new code.  That is what :data:`TRANSFORMS` is for.

**A line.**  Castagna's mudrock relation, for the case where all you will
admit to knowing is that the rock is clastic.

**The well's own trend.**  Where a well has shear sonic over *part* of its
interval, :func:`fit_vp_vs_trend` regresses Vs on Vp over that part.  Tempting
to assume this wins — it is this well's own rock rather than somebody else's —
but measured on 15/9-19-A it does not, and it is worth knowing why: a single
Vp-Vs line has nowhere to put shale volume, and shale volume is most of what
moves Vs.  Blind-tested on a held-out half of that well it reaches 6.2% median
error against 4.2% for the lithology-aware transform, and calibrated on the
shale-rich upper half and extrapolated into the sandier lower half it collapses
to 11% with an 11% low bias.  It earns its place as a *check* on a published
transform and as the fallback when no VSH exists — not as the default.

:func:`prediction_quality` measures any of the three against whatever measured
Vs the well has, so the choice is made on this well's evidence rather than on
habit or on the paragraph above.
"""

from __future__ import annotations

import numpy as np

from avo_qi.core.rockphysics import _GC_COEFFS

__all__ = [
    "GREENBERG_CASTAGNA",
    "TRANSFORMS",
    "MUDROCK_SLOPE_M_S",
    "MUDROCK_INTERCEPT_M_S",
    "lithology_fractions_from_vsh",
    "polynomial_vs",
    "mudrock_vs",
    "fit_vp_vs_trend",
    "apply_vp_vs_trend",
    "prediction_quality",
    "fill_missing_vs",
]

#: Greenberg & Castagna (1992) Vp-Vs polynomials, highest power first, for
#: velocities in **km/s** — imported rather than retyped, so the trend lines the
#: Rock Physics page draws and the curve predicted here cannot come from two
#: different sets of numbers.  Published coefficients are exactly the kind of
#: constant that gets copied once and then corrected in one place only.
GREENBERG_CASTAGNA = {lith: tuple(coeffs)
                      for lith, coeffs in _GC_COEFFS.items()}

#: Named transforms the app can offer.  Each is a ``{lithology: coefficients}``
#: map in km/s, so adding a published set is a dictionary entry — the maths in
#: :func:`polynomial_vs` does not change.  Only sets whose coefficients can be
#: checked against their source belong here: a transform that silently carries
#: half-remembered numbers is worse than no transform, because every Vp/Vs,
#: Poisson, gradient and substitution downstream inherits them without a mark.
TRANSFORMS = {
    "greenberg_castagna": GREENBERG_CASTAGNA,
}

#: Castagna, Batzle & Eastwood (1985), in m/s: ``Vs = 0.8621 Vp - 1172.4``.
MUDROCK_SLOPE_M_S = 0.8621
MUDROCK_INTERCEPT_M_S = -1172.4


def _arr(x):
    return np.asarray(x, dtype=float)


def lithology_fractions_from_vsh(vsh, shale="shale", matrix="sandstone"):
    """Per-sample lithology fractions from a shale volume.

    The obvious mapping, made explicit because it is an assumption: shale
    volume is the shale fraction and the rest is the matrix lithology.  A
    carbonate section wants ``matrix="limestone"``, and a well that is really a
    three-mineral system wants fractions built somewhere else and passed
    straight to :func:`polynomial_vs`.

    Non-finite VSH gives non-finite fractions rather than a default, so a
    sample with no shale volume produces no prediction instead of a confident
    clean-sand one.
    """
    vsh = _arr(vsh)
    clipped = np.clip(vsh, 0.0, 1.0)
    blank = ~np.isfinite(vsh)
    shale_fraction = np.where(blank, np.nan, clipped)
    return {shale: shale_fraction, matrix: 1.0 - shale_fraction}


def polynomial_vs(vp, fractions=None, coefficients=None):
    """Vs from a lithology-keyed Vp-Vs polynomial, Hill-averaged over the mix.

    The Greenberg-Castagna construction: evaluate each lithology's polynomial,
    then take the mean of the arithmetic and harmonic averages weighted by
    volume — the Hill average, which sits between the Voigt and Reuss bounds of
    the same mixture.

    Parameters
    ----------
    vp : array_like
        P velocity in **m/s**.
    fractions : dict, optional
        ``{lithology: fraction}``.  Each value may be a scalar or a per-sample
        array, so a VSH log drives it directly (see
        :func:`lithology_fractions_from_vsh`).  Defaults to pure sandstone.
    coefficients : dict, optional
        ``{lithology: polynomial}`` in km/s, highest power first.  Defaults to
        :data:`GREENBERG_CASTAGNA`; pass another published set to use it.

    Returns
    -------
    ndarray
        Vs in m/s, NaN wherever Vp or the fractions are not finite.
    """
    vp = _arr(vp)
    coefficients = dict(coefficients or GREENBERG_CASTAGNA)
    fractions = dict(fractions or {"sandstone": 1.0})

    unknown = set(fractions) - set(coefficients)
    if unknown:
        raise ValueError(
            "no coefficients for lithology/-ies: %s" % sorted(unknown))

    weights = {k: np.broadcast_to(_arr(v), vp.shape).astype(float)
               for k, v in fractions.items()}
    total = sum(weights.values())
    with np.errstate(invalid="ignore"):
        usable = np.isfinite(vp) & np.isfinite(total) & (total > 0)

    vp_km = np.where(usable, vp / 1000.0, np.nan)
    arithmetic = np.zeros(vp.shape)
    harmonic = np.zeros(vp.shape)
    for lith, weight in weights.items():
        share = np.where(usable, weight / total, np.nan)
        with np.errstate(invalid="ignore"):
            vs_km = np.maximum(np.polyval(coefficients[lith], vp_km), 1e-6)
        arithmetic += share * vs_km
        harmonic += share / vs_km

    with np.errstate(divide="ignore", invalid="ignore"):
        hill = 0.5 * (arithmetic + 1.0 / harmonic)
    return np.where(usable, hill * 1000.0, np.nan)


def mudrock_vs(vp):
    """Castagna's mudrock line, Vp and Vs both in m/s.

    One line for every clastic rock, which is its virtue and its limit: it
    knows nothing about shale volume, so it cannot separate a clean sand from
    the shale above it — the very contrast an AVO gradient is made of.
    """
    vp = _arr(vp)
    return np.where(np.isfinite(vp),
                    MUDROCK_SLOPE_M_S * vp + MUDROCK_INTERCEPT_M_S, np.nan)


def fit_vp_vs_trend(vp, vs, degree=1, min_samples=30):
    """Fit this well's own Vp-Vs trend to the interval that has both.

    A regression through a few hundred metres of *this* well, which sounds
    like it must beat a published transform and generally does not: the line
    has nowhere to put shale volume, so it fits the average lithology of its
    calibration interval and predicts that lithology everywhere.  **It is only
    as good as that interval is representative.**  On 15/9-19-A, calibrated on
    a random half it reaches 6.2% median error; calibrated on the shale-rich
    upper half and extrapolated into the sandier lower half — the natural thing
    to do when the shear sonic starts partway down — it reaches 11%, low by
    11%, because it is predicting shale velocities for sand.

    Use it as a check on a published transform, as the fallback where no VSH
    exists, and with the calibration interval in mind.  Raising ``degree``
    curves the line; it does not teach it about lithology.

    The fit is done in km/s so a quadratic is not conditioned on numbers of
    order 10^7.

    Returns
    -------
    dict
        ``coefficients`` (km/s, highest power first, ready for
        :func:`apply_vp_vs_trend`), ``n`` samples used, ``degree``, and
        ``reason`` where no fit was possible.
    """
    vp, vs = _arr(vp).ravel(), _arr(vs).ravel()
    if vp.size != vs.size:
        raise ValueError("vp and vs must be the same length")

    good = np.isfinite(vp) & np.isfinite(vs) & (vp > 0) & (vs > 0)
    blank = {"coefficients": None, "n": int(good.sum()), "degree": int(degree)}
    if good.sum() < int(min_samples):
        return {**blank, "reason":
                "only %d samples carry both Vp and Vs; %d are needed"
                % (good.sum(), int(min_samples))}
    if np.ptp(vp[good]) <= 0:
        return {**blank, "reason": "Vp does not vary over the fitted interval"}

    coefficients = np.polyfit(vp[good] / 1000.0, vs[good] / 1000.0, int(degree))
    return {"coefficients": tuple(float(c) for c in coefficients),
            "n": int(good.sum()), "degree": int(degree), "reason": None}


def apply_vp_vs_trend(vp, coefficients):
    """Evaluate a fitted trend. ``coefficients`` are km/s; Vp and Vs are m/s."""
    vp = _arr(vp)
    with np.errstate(invalid="ignore"):
        vs = np.polyval(np.asarray(coefficients, dtype=float), vp / 1000.0)
    return np.where(np.isfinite(vp) & (vs > 0), vs * 1000.0, np.nan)


def prediction_quality(measured, predicted):
    """Score a prediction against the measured Vs, where there is any.

    The number that decides which model to use, and the reason the panel offers
    a choice rather than a default.  ``bias`` is signed on purpose: a
    prediction that is 4% slow everywhere is a different problem from one that
    scatters, and only the first can be corrected.

    Returns
    -------
    dict
        ``n``, ``median_absolute_error`` and ``rms_error`` in m/s,
        ``median_relative_error`` and ``bias`` as fractions, and ``correlation``
        — all NaN where fewer than two samples can be compared.
    """
    measured, predicted = _arr(measured).ravel(), _arr(predicted).ravel()
    if measured.size != predicted.size:
        raise ValueError("measured and predicted must be the same length")

    good = np.isfinite(measured) & np.isfinite(predicted) & (measured > 0)
    blank = {"n": int(good.sum()), "median_absolute_error": np.nan,
             "rms_error": np.nan, "median_relative_error": np.nan,
             "bias": np.nan, "correlation": np.nan}
    if good.sum() < 2:
        return blank

    left, right = measured[good], predicted[good]
    residual = right - left
    correlation = (float(np.corrcoef(left, right)[0, 1])
                   if np.ptp(left) > 0 and np.ptp(right) > 0 else np.nan)
    return {"n": int(good.sum()),
            "median_absolute_error": float(np.median(np.abs(residual))),
            "rms_error": float(np.sqrt(np.mean(residual ** 2))),
            "median_relative_error": float(np.median(np.abs(residual) / left)),
            "bias": float(np.median(residual / left)),
            "correlation": correlation}


def fill_missing_vs(measured, predicted):
    """Keep every measured sample; fill only the gaps.

    A measured shear sonic beats any model of one, so the prediction is never
    allowed to overwrite it — it fills what is absent and nothing else.  The
    mask that comes back is the provenance the rest of the toolkit labels the
    curve with, and it is per sample rather than per well because a partially
    logged well is the common case.

    Returns
    -------
    dict
        ``vs`` (the merged curve) and ``is_predicted`` (True where the value
        came from the model).
    """
    measured, predicted = _arr(measured), _arr(predicted)
    if measured.shape != predicted.shape:
        raise ValueError("measured and predicted must be the same shape")

    gap = ~np.isfinite(measured)
    filled = np.where(gap, predicted, measured)
    return {"vs": filled, "is_predicted": gap & np.isfinite(predicted)}
