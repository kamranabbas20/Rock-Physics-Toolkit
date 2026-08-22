"""Gassmann fluid substitution.

Gassmann relates the bulk modulus of a rock saturated with one fluid to the
same rock saturated with another, through the dry frame that both share::

    K_sat / (K_min - K_sat) = K_dry / (K_min - K_dry)
                              + K_fl / (phi * (K_min - K_fl))

The shear modulus does not appear.  That is not an omission — it is the
theory's central claim: a fluid supports no shear, so mu is the same wet or
dry.  Every substitution here therefore carries mu through untouched, and
:func:`substitute` checks that it really did.

**Failure is informative, so it is reported rather than hidden.**  Running the
relation backwards from a measured log routinely produces a *negative* dry
frame, which does not mean "no answer" — it means the porosity, the mineral or
the measured velocities disagree with each other, because no real rock has a
dry frame softer than a vacuum.  This project met exactly that case: the demo
well's brine sand at 2500 / 1450 / 2.30 and phi 0.26 returns K_dry = -1.6 GPa
against quartz and brine, which is what exposed its porosity as invented.  A
silent NaN would have buried the finding, so every routine here returns the
values *and* a per-sample reason for each one it could not compute.

Moduli in GPa, densities in g/cc, velocities in m/s, porosity as a fraction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from avo_qi.core.rockphysics import (
    bulk_modulus,
    shear_modulus,
    velocities_from_moduli,
)

__all__ = [
    "GassmannResult",
    "gassmann_saturate",
    "gassmann_dry",
    "substitute",
]


@dataclass(frozen=True)
class GassmannResult:
    """Values with a per-sample account of where they could not be computed.

    ``values`` carries NaN wherever ``valid`` is False, and ``reasons`` says
    why in words at that sample ("" where the sample is fine).
    """

    values: np.ndarray
    valid: np.ndarray
    reasons: np.ndarray

    @property
    def n_invalid(self):
        return int((~self.valid).sum())

    def summary(self):
        """``{reason: count}`` over the samples that failed, commonest first."""
        bad = self.reasons[~self.valid]
        if bad.size == 0:
            return {}
        names, counts = np.unique(bad.astype(str), return_counts=True)
        order = np.argsort(-counts)
        return {str(names[i]): int(counts[i]) for i in order}

    def __array__(self, dtype=None):
        return np.asarray(self.values, dtype=dtype)


def _broadcast(*arrays):
    return np.broadcast_arrays(*[np.asarray(a, dtype=float) for a in arrays])


def _fail(reasons, valid, condition, message):
    """Record ``message`` where ``condition`` holds and the sample is still valid.

    First failure wins, so the checks read in the order that explains the
    sample best: a bad porosity is the reason, not the negative modulus it
    goes on to produce.
    """
    hit = condition & valid
    if hit.any():
        reasons[hit] = message
        valid[hit] = False
    return valid


def _common_checks(k_mineral, k_fluid, porosity, reasons, valid):
    valid = _fail(reasons, valid, ~np.isfinite(porosity), "porosity is not finite")
    valid = _fail(reasons, valid, (porosity <= 0.0) | (porosity >= 1.0),
                  "porosity outside (0, 1)")
    valid = _fail(reasons, valid, ~np.isfinite(k_mineral) | (k_mineral <= 0.0),
                  "mineral modulus is not positive")
    valid = _fail(reasons, valid, ~np.isfinite(k_fluid) | (k_fluid < 0.0),
                  "fluid modulus is negative")
    valid = _fail(reasons, valid, k_fluid >= k_mineral,
                  "fluid is stiffer than the mineral")
    return valid


def _solve(ratio, k_mineral, valid):
    """``K = K_min * r / (1 + r)`` from the Gassmann ratio, NaN where invalid."""
    out = np.full(ratio.shape, np.nan, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        candidate = k_mineral * ratio / (1.0 + ratio)
    np.copyto(out, candidate, where=valid)
    return out


def gassmann_saturate(k_dry, k_mineral, k_fluid, porosity):
    """Saturated bulk modulus of a dry frame filled with a fluid.

    Parameters
    ----------
    k_dry, k_mineral, k_fluid : array_like
        Dry-frame, mineral and pore-fluid bulk moduli in GPa.  Any of them may
        vary sample by sample; shapes are broadcast together.
    porosity : array_like
        Pore volume fraction.

    Returns
    -------
    GassmannResult
        ``values`` are the saturated bulk moduli in GPa.
    """
    k_dry, k_mineral, k_fluid, porosity = _broadcast(
        k_dry, k_mineral, k_fluid, porosity)
    reasons = np.full(k_dry.shape, "", dtype=object)
    valid = np.ones(k_dry.shape, dtype=bool)

    valid = _common_checks(k_mineral, k_fluid, porosity, reasons, valid)
    valid = _fail(reasons, valid, ~np.isfinite(k_dry), "dry modulus is not finite")
    valid = _fail(reasons, valid, k_dry < 0.0, "dry-frame modulus is negative")
    valid = _fail(reasons, valid, k_dry >= k_mineral,
                  "dry frame is stiffer than the mineral")

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (k_dry / (k_mineral - k_dry)
                 + k_fluid / (porosity * (k_mineral - k_fluid)))
    return GassmannResult(_solve(ratio, k_mineral, valid), valid, reasons)


def gassmann_dry(k_sat, k_mineral, k_fluid, porosity):
    """Dry-frame bulk modulus behind a saturated one — Gassmann run backwards.

    This is the half that fails on real data.  A negative result is reported
    as ``"dry-frame modulus is negative"`` rather than returned, because it is
    a statement about the *inputs*: that rock cannot be that mineral at that
    porosity with that fluid.

    Returns
    -------
    GassmannResult
        ``values`` are the dry-frame bulk moduli in GPa.
    """
    k_sat, k_mineral, k_fluid, porosity = _broadcast(
        k_sat, k_mineral, k_fluid, porosity)
    reasons = np.full(k_sat.shape, "", dtype=object)
    valid = np.ones(k_sat.shape, dtype=bool)

    valid = _common_checks(k_mineral, k_fluid, porosity, reasons, valid)
    valid = _fail(reasons, valid, ~np.isfinite(k_sat),
                  "saturated modulus is not finite")
    valid = _fail(reasons, valid, k_sat <= 0.0, "saturated modulus is not positive")
    valid = _fail(reasons, valid, k_sat >= k_mineral,
                  "saturated rock is as stiff as the mineral")

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (k_sat / (k_mineral - k_sat)
                 - k_fluid / (porosity * (k_mineral - k_fluid)))
    values = _solve(ratio, k_mineral, valid)

    # The physical failure, checked on the answer rather than the inputs.
    valid = _fail(reasons, valid, np.isfinite(values) & (values < 0.0),
                  "dry-frame modulus is negative")
    values = np.where(valid, values, np.nan)
    return GassmannResult(values, valid, reasons)


def substitute(vp, vs, rho, porosity, k_mineral, fluid_in, fluid_out):
    """Swap the pore fluid of a measured log.

    Runs Gassmann backwards to the dry frame with ``fluid_in``, then forwards
    with ``fluid_out``.  Density moves by mass balance on the *measured*
    density, so whatever matrix the rock actually has is preserved rather than
    replaced by an assumed mineral density.

    Parameters
    ----------
    vp, vs, rho : array_like
        Measured logs — m/s, m/s and g/cc.
    porosity : array_like
        Pore volume fraction.
    k_mineral : array_like
        Mineral bulk modulus in GPa, scalar or per sample.
    fluid_in, fluid_out : tuple
        ``(K in GPa, rho in g/cc)`` of the fluid in the rock now and of the
        one to put in its place.  Either element may be an array rather than a
        constant, which is how Batzle-Wang properties following a pressure and
        temperature profile are carried down the well.

    Returns
    -------
    dict
        ``VP``, ``VS``, ``RHOB`` (substituted logs, NaN where Gassmann could
        not be applied), ``K_dry``, ``valid`` and ``reasons``.
    """
    vp, vs, rho, porosity = _broadcast(vp, vs, rho, porosity)
    k_fluid_in, rho_fluid_in = (np.asarray(x, dtype=float) for x in fluid_in)
    k_fluid_out, rho_fluid_out = (np.asarray(x, dtype=float) for x in fluid_out)

    k_sat_in = bulk_modulus(vp, vs, rho)
    mu = shear_modulus(vs, rho)

    dry = gassmann_dry(k_sat_in, k_mineral, k_fluid_in, porosity)
    wet = gassmann_saturate(dry.values, k_mineral, k_fluid_out, porosity)

    # A sample that failed on the way down keeps *that* reason, which is the
    # one that explains it; only samples that survived can fail on the way up.
    valid = dry.valid & wet.valid
    reasons = np.where(dry.valid, wet.reasons, dry.reasons)

    rho_out = rho + porosity * (rho_fluid_out - rho_fluid_in)
    vp_out, vs_out = velocities_from_moduli(wet.values, mu, rho_out)

    vp_out = np.where(valid, vp_out, np.nan)
    vs_out = np.where(valid, vs_out, np.nan)
    rho_out = np.where(valid, rho_out, np.nan)

    _check_shear_invariance(vs_out, rho_out, mu, valid)

    return {
        "VP": vp_out,
        "VS": vs_out,
        "RHOB": rho_out,
        "K_dry": dry.values,
        "valid": valid,
        "reasons": reasons,
    }


def _check_shear_invariance(vs_out, rho_out, mu_in, valid, tol=1e-9):
    """Gassmann leaves mu alone; verify the output really carries it through.

    A class invariant rather than a data check — if this trips, the code is
    wrong, not the well.
    """
    check = valid & np.isfinite(vs_out) & np.isfinite(rho_out) & (mu_in > 0)
    if not check.any():
        return
    mu_out = shear_modulus(vs_out, rho_out)
    worst = float(np.max(np.abs(mu_out[check] - mu_in[check]) / mu_in[check]))
    if worst > tol:
        raise AssertionError(
            f"fluid substitution changed the shear modulus by {worst:.2e} "
            "relative; Gassmann requires it to be unchanged"
        )
