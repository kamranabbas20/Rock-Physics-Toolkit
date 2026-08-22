"""Measuring how far a well sits from a model.

An overlay drawn on a crossplot is a decoration until someone says by how much
the data misses it.  The eye is bad at this — a cloud of ten thousand samples
looks like it straddles a bound whether 99% or 60% of it is inside — and the
number is what turns the plot into a diagnostic.

Two questions, two functions:

:func:`bounds_check`
    *Is this sample physically possible?*  A point below the suspension bound
    cannot be that mineral with that fluid at that porosity, full stop.  This
    is the check that exposed the demo well's invented porosity, so it returns
    the offending **indices** and not merely a count — the samples that fail
    are the ones worth looking at.

:func:`log_misfit`
    *How far is the prediction from the log?*  Reported as a median bias and a
    robust scatter rather than a mean and a standard deviation, so that a
    handful of washed-out samples do not swamp the answer.

Nothing here fabricates a result for an empty selection: statistics of no
samples are NaN, and ``n`` says so.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["BoundsCheck", "LogMisfit", "bounds_check", "log_misfit", "curve_misfit"]

_NAN = float("nan")


def _finite(*arrays):
    mask = np.ones(np.asarray(arrays[0], float).shape, dtype=bool)
    for a in arrays:
        mask &= np.isfinite(np.asarray(a, dtype=float))
    return mask


def _percent(fraction):
    return "n/a" if not np.isfinite(fraction) else f"{100.0 * fraction:.0f}%"


@dataclass(frozen=True)
class BoundsCheck:
    """How much of a selection lies between a pair of bounds."""

    n: int
    n_inside: int
    fraction_inside: float
    outside: np.ndarray
    below: np.ndarray
    above: np.ndarray
    worst_breach: float
    worst_index: int

    def describe(self, unit="", depths=None):
        """One sentence for a page to show.

        ``depths``, when given, names where the worst breach happens, which is
        the first thing anyone asks next.
        """
        if self.n == 0:
            return "No samples with both a value and a bound to compare it with."
        head = (f"{_percent(self.fraction_inside)} of {self.n:,} samples lie "
                f"inside the bounds.")
        if self.n_inside == self.n:
            return head + " Nothing falls outside."
        where = ""
        if depths is not None and 0 <= self.worst_index < len(np.asarray(depths)):
            where = f" at {float(np.asarray(depths)[self.worst_index]):.1f} m"
        side = "below the lower" if self.worst_breach < 0 else "above the upper"
        suffix = f" {unit}" if unit else ""
        return (f"{head} The worst breach is {abs(self.worst_breach):.3g}{suffix} "
                f"{side} bound{where}.")


def bounds_check(values, lower, upper):
    """Which samples fall outside a pair of bounds, and by how much.

    Parameters
    ----------
    values : array_like
        The measured quantity.
    lower, upper : array_like
        The bounds, either scalars or one per sample.  They are ordered
        internally, so passing them the wrong way round is not an error.

    Returns
    -------
    BoundsCheck
        ``outside`` holds indices into the original arrays, so a caller can
        colour exactly those samples without re-deriving anything.
    """
    v, lo, hi = np.broadcast_arrays(
        np.asarray(values, dtype=float),
        np.asarray(lower, dtype=float),
        np.asarray(upper, dtype=float),
    )
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)

    good = _finite(v, lo, hi)
    n = int(good.sum())
    empty = np.array([], dtype=int)
    if n == 0:
        false = np.zeros(v.shape, dtype=bool)
        return BoundsCheck(0, 0, _NAN, empty, false, false, _NAN, -1)

    below = good & (v < lo)
    above = good & (v > hi)
    outside_mask = below | above
    n_inside = n - int(outside_mask.sum())

    breach = np.zeros(v.shape, dtype=float)
    breach[below] = (v - lo)[below]          # negative: short of the lower bound
    breach[above] = (v - hi)[above]          # positive: past the upper bound
    worst_index = int(np.argmax(np.abs(breach))) if outside_mask.any() else -1
    worst = float(breach[worst_index]) if worst_index >= 0 else 0.0

    return BoundsCheck(
        n=n, n_inside=n_inside, fraction_inside=n_inside / n,
        outside=np.flatnonzero(outside_mask), below=below, above=above,
        worst_breach=worst, worst_index=worst_index,
    )


@dataclass(frozen=True)
class LogMisfit:
    """How far a predicted log sits from the measured one."""

    n: int
    bias: float
    scatter: float
    rms: float
    relative_bias: float
    correlation: float
    worst: float
    worst_index: int

    def describe(self, name="", unit="", depths=None):
        """One sentence for a page to show."""
        label = f"{name}: " if name else ""
        if self.n == 0:
            return f"{label}nothing to compare — no sample has both a prediction and a log."
        suffix = f" {unit}" if unit else ""
        sign = "high" if self.bias >= 0 else "low"
        where = ""
        if depths is not None and 0 <= self.worst_index < len(np.asarray(depths)):
            where = f" at {float(np.asarray(depths)[self.worst_index]):.1f} m"
        return (
            f"{label}the model runs {abs(self.bias):.4g}{suffix} {sign} "
            f"({abs(self.relative_bias) * 100.0:.1f}%), with a scatter of "
            f"{self.scatter:.4g}{suffix} over {self.n:,} samples. "
            f"Worst sample {self.worst:+.4g}{suffix}{where}."
        )


def _misfit_from_residual(residual, measured, index):
    n = residual.size
    if n == 0:
        return LogMisfit(0, _NAN, _NAN, _NAN, _NAN, _NAN, _NAN, -1)

    bias = float(np.median(residual))
    # Median absolute deviation about the median — robust where a standard
    # deviation would be dragged around by a few bad samples.
    scatter = float(np.median(np.abs(residual - bias)))
    rms = float(np.sqrt(np.mean(residual ** 2)))
    scale = float(np.median(np.abs(measured)))
    relative = bias / scale if scale > 0 else _NAN

    correlation = _NAN
    if n >= 2 and np.std(measured) > 0 and np.std(measured + residual) > 0:
        correlation = float(np.corrcoef(measured, measured + residual)[0, 1])

    worst_at = int(np.argmax(np.abs(residual)))
    return LogMisfit(n, bias, scatter, rms, relative, correlation,
                     float(residual[worst_at]), int(index[worst_at]))


def log_misfit(predicted, measured):
    """Compare a predicted log with the measured one, sample by sample.

    ``bias`` is the median of ``predicted - measured``: positive means the
    model runs high.  ``worst_index`` indexes the *original* arrays, so a
    caller can find the depth it happened at.
    """
    p = np.asarray(predicted, dtype=float).ravel()
    m = np.asarray(measured, dtype=float).ravel()
    if p.shape != m.shape:
        raise ValueError("predicted and measured must have the same length")
    good = _finite(p, m)
    index = np.flatnonzero(good)
    return _misfit_from_residual(p[good] - m[good], m[good], index)


def curve_misfit(x, y, model_x, model_y):
    """Compare data with a model *curve* by interpolating it onto the data's x.

    Samples whose ``x`` falls outside the curve's range are dropped rather than
    compared against a clamped end point, which would report a fit where there
    is no model to fit to.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    mx = np.asarray(model_x, dtype=float).ravel()
    my = np.asarray(model_y, dtype=float).ravel()
    if x.shape != y.shape:
        raise ValueError("x and y must have the same length")
    if mx.shape != my.shape:
        raise ValueError("the model's x and y must have the same length")

    model_good = _finite(mx, my)
    if model_good.sum() < 2:
        return LogMisfit(0, _NAN, _NAN, _NAN, _NAN, _NAN, _NAN, -1)
    mx, my = mx[model_good], my[model_good]
    order = np.argsort(mx)
    mx, my = mx[order], my[order]

    good = _finite(x, y) & (x >= mx[0]) & (x <= mx[-1])
    index = np.flatnonzero(good)
    if index.size == 0:
        return LogMisfit(0, _NAN, _NAN, _NAN, _NAN, _NAN, _NAN, -1)
    predicted = np.interp(x[good], mx, my)
    return _misfit_from_residual(predicted - y[good], y[good], index)
