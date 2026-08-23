"""Monte Carlo uncertainty on the rock physics forward model.

Every number the forward model produces is a point answer standing on inputs
that are not points.  VSH is good to a few percent at best, a saturation from
resistivity to rather less than that, and the critical porosity and effective
pressure a frame model needs are estimates rather than measurements.  A single
predicted Vp hides all of it, and — worse — an AVO class label hides it
completely: "Class III" reads as a fact when it may be a coin toss against
Class IIn.

This module samples the inputs and runs the model many times, so the answers
come back as distributions:

* :func:`monte_carlo_forward` — predicted Vp, Vs and RHOB as realisations,
  summarised with :func:`percentile_bands` into P10-P50-P90.
* :func:`class_probabilities` — the payoff.  Per reflector, the *probability*
  of each AVO class rather than one label, plus how confident that is.

Two kinds of uncertainty are kept apart, because conflating them is wrong:

**Log noise** is per sample.  Each depth's VSH carries its own measurement
error, and neighbouring samples are not obliged to err the same way.

**Model parameters** are per realisation.  A critical porosity is a property
of the rock type, not of the sample; drawing a fresh one at every depth would
model a well whose grain packing changes every 15 cm, which is not the
uncertainty anyone means.

Correlations between log noises default to **zero**.  VSH and porosity really
are anti-correlated in most clastics, and saying so narrows the cloud
considerably — but that is an assumption about the rock, so it has to be
stated rather than assumed on the reader's behalf.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from avo_qi.core.avo import CLASSES, classify_array
from avo_qi.core.petro import forward_model

__all__ = [
    "Prior",
    "LogNoise",
    "MonteCarloResult",
    "DEFAULT_LOG_NOISE",
    "DEFAULT_PRIORS",
    "PRIOR_PARAMETERS",
    "monte_carlo_forward",
    "perturb_logs",
    "percentile_bands",
    "class_probabilities",
    "class_probabilities_from_layers",
    "batched_aki_richards",
    "batched_shuey_fit",
]

_INF = float("inf")

#: Parameters of :func:`avo_qi.core.petro.forward_model` a prior may be put on.
#: Deliberately a closed set — a typo in a key would otherwise be sampled,
#: discarded and never noticed.
PRIOR_PARAMETERS = ("phi_c", "n_grains", "pressure", "shear_factor",
                    "brie_exponent")


# ------------------------------------------------------------- priors ------
@dataclass(frozen=True)
class Prior:
    """A distribution for one model parameter, drawn once per realisation.

    Build these with the constructors rather than the raw fields::

        Prior.normal(0.36, 0.03, low=0.20, high=0.50)
        Prior.uniform(0.5, 1.0)
        Prior.triangular(10e6, 20e6, 35e6)
        Prior.fixed(0.36)
    """

    kind: str
    params: tuple
    low: float = -_INF
    high: float = _INF

    @classmethod
    def normal(cls, mean, sd, low=-_INF, high=_INF):
        if float(sd) < 0:
            raise ValueError("a standard deviation cannot be negative")
        return cls("normal", (float(mean), float(sd)), float(low), float(high))

    @classmethod
    def lognormal(cls, median, sigma, low=-_INF, high=_INF):
        """Multiplicative spread — for a parameter that cannot go negative."""
        if float(median) <= 0:
            raise ValueError("a lognormal median must be positive")
        return cls("lognormal", (float(median), float(sigma)),
                   float(low), float(high))

    @classmethod
    def uniform(cls, low, high):
        if float(high) < float(low):
            low, high = high, low
        return cls("uniform", (float(low), float(high)), float(low), float(high))

    @classmethod
    def triangular(cls, low, mode, high):
        if not (float(low) <= float(mode) <= float(high)):
            raise ValueError("a triangular prior needs low <= mode <= high")
        return cls("triangular", (float(low), float(mode), float(high)),
                   float(low), float(high))

    @classmethod
    def fixed(cls, value):
        """No uncertainty — useful to pin one parameter while varying others."""
        return cls("fixed", (float(value),), float(value), float(value))

    @property
    def is_fixed(self):
        return self.kind == "fixed" or (
            self.kind == "normal" and self.params[1] == 0.0)

    def draw(self, rng, size):
        """``size`` samples, truncated to ``[low, high]`` by clipping."""
        if self.kind == "fixed":
            out = np.full(size, self.params[0], dtype=float)
        elif self.kind == "normal":
            out = rng.normal(self.params[0], self.params[1], size)
        elif self.kind == "lognormal":
            out = self.params[0] * np.exp(rng.normal(0.0, self.params[1], size))
        elif self.kind == "uniform":
            out = rng.uniform(self.params[0], self.params[1], size)
        elif self.kind == "triangular":
            out = rng.triangular(*self.params, size)
        else:
            raise ValueError(f"unknown prior kind {self.kind!r}")
        return np.clip(out, self.low, self.high)

    def describe(self):
        if self.kind == "fixed":
            return f"fixed at {self.params[0]:g}"
        if self.kind == "normal":
            return f"normal, mean {self.params[0]:g} sd {self.params[1]:g}"
        if self.kind == "lognormal":
            return f"lognormal, median {self.params[0]:g} sigma {self.params[1]:g}"
        if self.kind == "uniform":
            return f"uniform {self.params[0]:g} to {self.params[1]:g}"
        return (f"triangular {self.params[0]:g} / {self.params[1]:g} / "
                f"{self.params[2]:g}")


@dataclass(frozen=True)
class LogNoise:
    """Measurement uncertainty on a petrophysical log, added per sample.

    ``sigma`` is in the log's own units — 0.05 on VSH means five saturation
    units of shale volume.  ``low`` and ``high`` clip the perturbed log back
    into its physical range, since a fraction cannot leave [0, 1].
    """

    sigma: float
    low: float = 0.0
    high: float = 1.0

    @property
    def is_zero(self):
        return float(self.sigma) == 0.0


#: Defaults that are honest rather than flattering.  A shale volume from GR is
#: good to a few percent; a saturation from resistivity is the least certain
#: number in the whole chain, and its spread dominates a gas-sand answer.
DEFAULT_LOG_NOISE = {
    "VSH": LogNoise(0.05),
    "PHIT": LogNoise(0.02),
    "SW": LogNoise(0.10),
}

#: Standalone defaults for the frame parameters, for headless use.  A caller
#: with its own settings should centre its priors on those instead.
DEFAULT_PRIORS = {
    "phi_c": Prior.normal(0.36, 0.03, low=0.20, high=0.50),
    "n_grains": Prior.normal(9.0, 1.5, low=4.0, high=16.0),
    "pressure": Prior.normal(20e6, 5e6, low=1e6),
    "shear_factor": Prior.uniform(0.6, 1.0),
}


# ------------------------------------------------------------- sampling ----
def _correlated_normals(rng, names, correlation, shape):
    """Standard normals for each name, correlated where asked.

    ``correlation`` is ``{(name_a, name_b): rho}``.  Anything unmentioned is
    independent, which is the default for exactly the reason in the module
    docstring: a correlation is a claim about the rock.
    """
    n = len(names)
    raw = rng.standard_normal((n,) + tuple(shape))
    if not correlation:
        return dict(zip(names, raw))

    index = {name: i for i, name in enumerate(names)}
    matrix = np.eye(n)
    for (a, b), rho in correlation.items():
        if a not in index or b not in index:
            raise ValueError(
                f"correlation names {a!r}, {b!r} must both be among {list(names)}")
        rho = float(rho)
        if not -1.0 <= rho <= 1.0:
            raise ValueError(f"a correlation must lie in [-1, 1], got {rho}")
        matrix[index[a], index[b]] = matrix[index[b], index[a]] = rho

    try:
        factor = np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        raise ValueError(
            "those correlations are not mutually consistent — the correlation "
            "matrix is not positive definite"
        ) from None

    flat = raw.reshape(n, -1)
    return dict(zip(names, (factor @ flat).reshape(raw.shape)))


@dataclass(frozen=True)
class MonteCarloResult:
    """Realisations of the predicted logs, plus the draws behind them.

    ``VP``, ``VS`` and ``RHOB`` are ``(n_realisations, n_samples)``.
    """

    VP: np.ndarray
    VS: np.ndarray
    RHOB: np.ndarray
    draws: dict = field(default_factory=dict)
    n_realisations: int = 0
    n_failed: int = 0

    def curve(self, name):
        try:
            return getattr(self, str(name).upper())
        except AttributeError:
            raise ValueError(
                f"no curve {name!r}; have VP, VS and RHOB") from None

    def bands(self, name, percentiles=(10, 50, 90)):
        """Percentile bands down the well for one curve."""
        return percentile_bands(self.curve(name), percentiles)

    def spread(self, name, percentiles=(10, 90)):
        """Width of the band, sample by sample — the uncertainty in one array."""
        low, high = self.bands(name, percentiles).values()
        return high - low


def monte_carlo_forward(vsh, phit, sw, n_realisations=250, seed=0,
                        log_noise=None, priors=None, correlation=None,
                        **model_kwargs):
    """Run :func:`avo_qi.core.petro.forward_model` many times over its inputs.

    Parameters
    ----------
    vsh, phit, sw : array_like
        The petrophysical logs, as the deterministic model takes them.
    n_realisations : int
        How many times to run.  A few hundred is enough for P10-P90; the cost
        is linear and the model is vectorised over samples, so each realisation
        is one array pass.
    seed : int
        Fixes the draw.  The same seed gives the same answer, which is what
        makes a reported P10 checkable.
    log_noise : dict, optional
        ``{'VSH': LogNoise(...), 'PHIT': ..., 'SW': ...}``.  Defaults to
        :data:`DEFAULT_LOG_NOISE`; pass ``{}`` for noise-free logs.
    priors : dict, optional
        ``{parameter: Prior}`` over :data:`PRIOR_PARAMETERS`.  Each is drawn
        once per realisation and overrides the matching ``model_kwargs``.
        Defaults to none — the deterministic parameters are used as given.
    correlation : dict, optional
        ``{('VSH', 'PHIT'): -0.5}`` and so on, between the *log* noises.
    **model_kwargs
        Passed straight through to ``forward_model``: ``matrix``, ``shale``,
        ``frame``, ``hydrocarbon``, ``phi_c`` and the rest.

    Returns
    -------
    MonteCarloResult
    """
    vsh = np.atleast_1d(np.asarray(vsh, dtype=float))
    phit = np.atleast_1d(np.asarray(phit, dtype=float))
    sw = np.atleast_1d(np.asarray(sw, dtype=float))
    if not (vsh.shape == phit.shape == sw.shape):
        raise ValueError("VSH, PHIT and SW must have the same length")

    n_realisations = int(n_realisations)
    if n_realisations < 1:
        raise ValueError("n_realisations must be at least 1")

    log_noise = DEFAULT_LOG_NOISE if log_noise is None else dict(log_noise)
    priors = dict(priors or {})
    unknown = set(priors) - set(PRIOR_PARAMETERS)
    if unknown:
        raise ValueError(
            f"no prior can be put on {sorted(unknown)}; "
            f"expected some of {list(PRIOR_PARAMETERS)}")

    rng = np.random.default_rng(seed)
    logs = {"VSH": vsh, "PHIT": phit, "SW": sw}
    n_samples = vsh.size

    # Per-sample noise: one field of standard normals per noisy log.
    noisy = [name for name in ("VSH", "PHIT", "SW")
             if name in log_noise and not log_noise[name].is_zero]
    normals = _correlated_normals(rng, noisy, correlation,
                                  (n_realisations, n_samples)) if noisy else {}

    perturbed = {}
    for name, base in logs.items():
        if name in normals:
            spec = log_noise[name]
            perturbed[name] = np.clip(base + spec.sigma * normals[name],
                                      spec.low, spec.high)
        else:
            perturbed[name] = np.broadcast_to(base, (n_realisations, n_samples))

    # Per-realisation parameters: one draw each, not one per sample.
    draws = {name: prior.draw(rng, n_realisations)
             for name, prior in priors.items()}

    vp = np.empty((n_realisations, n_samples), dtype=float)
    vs = np.empty_like(vp)
    rho = np.empty_like(vp)
    n_failed = 0

    for r in range(n_realisations):
        kwargs = dict(model_kwargs)
        for name, values in draws.items():
            kwargs[name] = values[r]
        out = forward_model(perturbed["VSH"][r], perturbed["PHIT"][r],
                            perturbed["SW"][r], **kwargs)
        vp[r], vs[r], rho[r] = out["VP"], out["VS"], out["RHOB"]
        if not out["valid"].all():
            n_failed += 1

    for name, values in perturbed.items():
        draws.setdefault(name, values)

    return MonteCarloResult(VP=vp, VS=vs, RHOB=rho, draws=draws,
                            n_realisations=n_realisations, n_failed=n_failed)



def perturb_logs(vp, vs, rho, n_realisations=250, seed=0, vp_pct=1.0,
                 vs_pct=2.0, rho_pct=1.0, correlation=None):
    """Realisations of measured elastic logs, from their own measurement error.

    A different question from :func:`monte_carlo_forward`, and the one an AVO
    class actually turns on.  There the uncertainty is in a *model* of the
    rock; here the logs are taken as given and the doubt is in the
    measurement — how well the sonic and density tools read — which is what
    decides whether an intercept and gradient land firmly inside a class or on
    its boundary.

    Uncertainties are **relative**, as a percentage, because that is how log
    accuracy is quoted.  ``vs_pct`` defaults to twice ``vp_pct``: a shear
    sonic is the noisier measurement, and Vs carries the gradient.

    Returns ``{'VP': ..., 'VS': ..., 'RHOB': ...}``, each
    ``(n_realisations, n_samples)``.
    """
    vp = np.atleast_1d(np.asarray(vp, dtype=float))
    vs = np.atleast_1d(np.asarray(vs, dtype=float))
    rho = np.atleast_1d(np.asarray(rho, dtype=float))
    if not (vp.shape == vs.shape == rho.shape):
        raise ValueError("VP, VS and RHOB must have the same length")
    n_realisations = int(n_realisations)
    if n_realisations < 1:
        raise ValueError("n_realisations must be at least 1")

    rng = np.random.default_rng(seed)
    percentages = {"VP": float(vp_pct), "VS": float(vs_pct),
                   "RHOB": float(rho_pct)}
    if any(value < 0 for value in percentages.values()):
        raise ValueError("a percentage uncertainty cannot be negative")

    noisy = [name for name, value in percentages.items() if value > 0]
    normals = _correlated_normals(rng, noisy, correlation,
                                  (n_realisations, vp.size)) if noisy else {}

    out = {}
    for name, base in (("VP", vp), ("VS", vs), ("RHOB", rho)):
        if name in normals:
            scale = 1.0 + percentages[name] / 100.0 * normals[name]
            # A velocity or density cannot go negative, however bad the tool.
            out[name] = base * np.maximum(scale, 1e-6)
        else:
            out[name] = np.broadcast_to(base, (n_realisations, vp.size)).copy()
    return out


def percentile_bands(realisations, percentiles=(10, 50, 90)):
    """``{percentile: array}`` down the samples, ignoring failed realisations.

    A sample where every realisation failed comes back NaN rather than as a
    fabricated value.
    """
    values = np.asarray(realisations, dtype=float)
    if values.ndim != 2:
        raise ValueError("realisations must be (n_realisations, n_samples)")
    out = {}
    clean = np.where(np.isfinite(values), values, np.nan)
    all_nan = np.all(~np.isfinite(values), axis=0)
    for q in percentiles:
        # A sample no realisation could model is an expected outcome here, not
        # a problem, so its all-NaN slice is silenced rather than printed.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            band = np.nanpercentile(clean, float(q), axis=0)
        out[float(q)] = np.where(all_nan, np.nan, band)
    return out


# ------------------------------------------------- AVO class probability ---
def batched_aki_richards(vp1, vs1, rho1, vp2, vs2, rho2, angles):
    """Aki-Richards over whole arrays of interfaces at once.

    Same physics as :func:`avo_qi.core.reflectivity.aki_richards_rpp`, which
    takes one interface at a time.  Monte Carlo needs tens of thousands of
    interfaces per run, so this broadcasts instead of looping; the two are
    pinned against each other in the tests rather than trusted to agree.

    The layer arrays broadcast together, and ``angles`` is appended as a
    trailing axis: ``(..., n_angles)``.
    """
    vp1, vs1, rho1, vp2, vs2, rho2 = np.broadcast_arrays(
        *[np.asarray(a, dtype=float)
          for a in (vp1, vs1, rho1, vp2, vs2, rho2)])
    theta = np.radians(np.asarray(angles, dtype=float))

    vp = (vp1 + vp2) / 2.0
    vs = (vs1 + vs2) / 2.0
    rho = (rho1 + rho2) / 2.0
    dvp, dvs, drho = vp2 - vp1, vs2 - vs1, rho2 - rho1

    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(vp != 0, (vs / vp) ** 2, 0.0)[..., np.newaxis]
        sin2 = np.sin(theta) ** 2
        sec2 = 1.0 / np.cos(theta) ** 2

        term_rho = 0.5 * (1 - 4 * k * sin2) * (drho / rho)[..., np.newaxis]
        term_vp = 0.5 * (dvp / vp)[..., np.newaxis] * sec2
        term_vs = -4 * k * sin2 * np.where(vs != 0, dvs / vs, 0.0)[..., np.newaxis]
    return term_rho + term_vp + term_vs


def batched_shuey_fit(rc, angles):
    """Two-term Shuey fit for many reflectors at once, NaN-safe.

    :func:`avo_qi.core.avo.shuey_fit` handles a blanked angle by falling back
    to a per-row ``lstsq``, which is right but carries the setup cost of a
    fresh SVD for every row.  Masking past the critical angle blanks *most*
    rows, and a Monte Carlo has tens of thousands of them, so that path costs
    tens of seconds where the whole model costs a tenth of one.

    A two-term fit does not need a solver.  ``R = A + B x`` has the closed
    form below, and dropping a blanked angle is just leaving it out of the
    sums — so the whole thing is a handful of array reductions and the answer
    is the same least-squares answer, not an approximation.  The tests pin it
    against ``shuey_fit`` to keep it that way.

    Returns ``(A, B)``, each shaped like ``rc`` without its angle axis.
    """
    rc = np.asarray(rc, dtype=float)
    x = np.sin(np.radians(np.asarray(angles, dtype=float))) ** 2
    if x.size != rc.shape[-1]:
        raise ValueError("angles must have one entry per reflectivity sample")

    good = np.isfinite(rc)
    y = np.where(good, rc, 0.0)
    w = good.astype(float)

    s0 = w.sum(axis=-1)
    s1 = (w * x).sum(axis=-1)
    s2 = (w * x * x).sum(axis=-1)
    sy = y.sum(axis=-1)
    sxy = (y * x).sum(axis=-1)

    det = s0 * s2 - s1 * s1
    with np.errstate(divide="ignore", invalid="ignore"):
        A = (s2 * sy - s1 * sxy) / det
        B = (s0 * sxy - s1 * sy) / det
    # Fewer than two surviving angles cannot define a line.
    undetermined = (s0 < 2) | ~np.isfinite(det) | (det == 0)
    return (np.where(undetermined, np.nan, A), np.where(undetermined, np.nan, B))


def _post_critical_mask(vp1, vp2, angles):
    """True where an angle is at or beyond the interface's critical angle."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where((vp2 > vp1) & (vp1 > 0), vp1 / vp2, np.nan)
        critical = np.degrees(np.arcsin(np.clip(ratio, -1.0, 1.0)))
    theta = np.asarray(angles, dtype=float)
    return np.isfinite(critical)[..., np.newaxis] & (theta >= critical[..., np.newaxis])


def class_probabilities(vp, vs, rho, samples, angles, a_tol=0.02,
                        mask_post_critical=True, classes=None):
    """Probability of each AVO class, per reflector, across realisations.

    Parameters
    ----------
    vp, vs, rho : ndarray
        ``(n_realisations, n_samples)`` realisations, from
        :func:`monte_carlo_forward`.
    samples : array_like
        Interface indices.  Interface ``i`` is between sample ``i`` and
        ``i + 1``, matching :func:`avo_qi.core.avo.reflector_avo`.
    angles : array_like
        Incidence angles in degrees.
    a_tol : float
        Intercept band separating Class II from I and III, as elsewhere.
    mask_post_critical : bool
        Blank angles past each realisation's own critical angle before
        fitting.  Past critical the real continuation spikes and drags the
        gradient the wrong way, which is enough to turn a Class I into
        background — and in a Monte Carlo it would show up as spurious
        class spread rather than as an obvious artefact.

    Returns
    -------
    pandas.DataFrame
        One row per interface, indexed by sample.  A column per AVO class
        holding its probability, plus ``modal_class``, ``confidence`` (that
        class's probability), ``n_valid``, and P10/P50/P90 of A and B.

    Notes
    -----
    Takes each interface as the pair of *adjacent samples* around it.  Where a
    caller classifies blocked layers instead, it must use
    :func:`class_probabilities_from_layers` with the same blocking.
    """
    import pandas as pd

    vp = np.atleast_2d(np.asarray(vp, dtype=float))
    vs = np.atleast_2d(np.asarray(vs, dtype=float))
    rho = np.atleast_2d(np.asarray(rho, dtype=float))
    idx = np.atleast_1d(np.asarray(samples, dtype=int))
    angles = np.atleast_1d(np.asarray(angles, dtype=float))
    classes = list(classes or CLASSES)

    if idx.size == 0:
        return pd.DataFrame(columns=["sample"] + classes +
                            ["modal_class", "confidence", "n_valid"])
    if idx.min() < 0 or idx.max() + 1 >= vp.shape[1]:
        raise ValueError("every interface needs a sample below it, so indices "
                         "must lie in [0, n_samples - 2]")

    upper, lower = idx, idx + 1
    return class_probabilities_from_layers(
        (vp[:, upper], vs[:, upper], rho[:, upper]),
        (vp[:, lower], vs[:, lower], rho[:, lower]),
        angles, samples=idx, a_tol=a_tol,
        mask_post_critical=mask_post_critical, classes=classes)


def class_probabilities_from_layers(upper, lower, angles, samples=None,
                                    a_tol=0.02, mask_post_critical=True,
                                    classes=None):
    """Class odds for interfaces whose two layers are given directly.

    The general form behind :func:`class_probabilities`.  Adjacent samples are
    only one way to define an interface: a page that classifies **blocked**
    layers — a half cycle averaged either side of the boundary — has to run its
    Monte Carlo on those same layers, or the odds describe a different
    interface from the label they are attached to and any disagreement between
    them is an artefact of the mismatch rather than a finding.

    Parameters
    ----------
    upper, lower : tuple
        ``(vp, vs, rho)`` for the layer above and below each interface, each
        ``(n_realisations, n_interfaces)``.
    samples : array_like, optional
        Interface indices, carried into the table for labelling.
    """
    import pandas as pd

    vp1, vs1, rho1 = (np.atleast_2d(np.asarray(a, dtype=float)) for a in upper)
    vp2, vs2, rho2 = (np.atleast_2d(np.asarray(a, dtype=float)) for a in lower)
    angles = np.atleast_1d(np.asarray(angles, dtype=float))
    classes = list(classes or CLASSES)
    if vp1.shape != vp2.shape:
        raise ValueError("the upper and lower layers must have the same shape")

    n_real, n_int = vp1.shape
    idx = (np.arange(n_int) if samples is None
           else np.atleast_1d(np.asarray(samples, dtype=int)))
    if idx.size != n_int:
        raise ValueError("samples must have one entry per interface")
    if n_int == 0:
        return pd.DataFrame(columns=["sample"] + classes +
                            ["modal_class", "confidence", "n_valid"])

    rc = batched_aki_richards(vp1, vs1, rho1, vp2, vs2, rho2, angles)
    if mask_post_critical:
        rc = np.where(_post_critical_mask(vp1, vp2, angles), np.nan, rc)

    A, B = batched_shuey_fit(rc, angles)
    labels = classify_array(A.ravel(), B.ravel(),
                            a_tol=a_tol).reshape(n_real, n_int)

    rows = []
    for j, sample in enumerate(idx):
        column = labels[:, j]
        finite = np.isfinite(A[:, j]) & np.isfinite(B[:, j])
        counts = {name: float(np.mean(column == name)) for name in classes}
        modal = max(counts, key=counts.get)
        row = {"sample": int(sample)}
        row.update(counts)
        row["modal_class"] = modal
        row["confidence"] = counts[modal]
        row["n_valid"] = int(finite.sum())
        for name, values in (("A", A[:, j]), ("B", B[:, j])):
            for q in (10, 50, 90):
                row[f"{name}_P{q}"] = (float(np.nanpercentile(values, q))
                                       if finite.any() else float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)
