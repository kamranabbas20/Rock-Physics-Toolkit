"""Transversely isotropic (VTI) reflectivity, after Rüger.

Every gradient the rest of this toolkit fits assumes the rock either side of
an interface is isotropic.  Shale is not.  A shale with a bedding-parallel
fabric is transversely isotropic about the vertical, and the consequence for
AVO is not subtle: the P-P gradient picks up the contrast in Thomsen's delta
directly, so a shale-over-sand top can move a whole class on anisotropy alone,
with no change in the rock or the fluid.

Rüger's (1997, 2002) weak-anisotropy, weak-contrast form is

    R(t) = 1/2 dZ/Z
         + 1/2 [ da/a - (2 b/a)^2 dG/G + d(delta) ] sin^2 t
         + 1/2 [ da/a + d(epsilon) ]              sin^2 t tan^2 t

with ``a`` the vertical P velocity, ``b`` the vertical S velocity,
``Z = rho a`` the P impedance, ``G = rho b^2`` the shear modulus, a bar
meaning the average across the interface and ``d`` the contrast (lower minus
upper).  Set both Thomsen parameters to zero on both sides and this is the
Aki-Richards three-term expression written in impedance variables.

**Where the anisotropy actually lands.**  Only the *contrasts* appear.  Two
shales with the same strong anisotropy give an isotropic-looking reflection;
an isotropic sand under an anisotropic shale gives the full effect.  Reading
the three terms off, an interface picks up

    A unchanged            — normal incidence never sees the anisotropy
    B + 1/2 d(delta)       — the gradient, which is what the class is made of
    C + 1/2 d(epsilon)     — the far-angle term, out of reach of most gathers

so a shale-over-sand top with a shale delta of 0.10 and an isotropic sand
carries a gradient shift of -0.05 (the contrast is sand minus shale).  That
is the same size as the whole Class IIn band, which is why this module
exists.

**Nothing here supplies epsilon and delta.**  They are a measurement — from
walkaway VSP, from cores, or from a local calibration — and inventing a
plausible-looking pair would put a number into a class label that no
measurement supports.  The defaults are zero everywhere, so a caller that
says nothing gets exactly the isotropic answer it gets today.
``LITERATURE_SHALES`` is offered as a starting range with its source named,
not as a default.

Unit conventions follow the rest of ``core/``: velocities in m/s, densities
in g/cc, angles in degrees.  Thomsen parameters are dimensionless.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "Thomsen",
    "ISOTROPIC",
    "LITERATURE_SHALES",
    "ruger_terms",
    "ruger_vti_rpp",
    "gradient_shift",
    "fitted_gradient_shift",
    "thomsen_from_vsh",
    "thomsen_logs_from_vsh",
]


@dataclass(frozen=True)
class Thomsen:
    """Thomsen's (1986) weak-anisotropy parameters for one medium.

    ``epsilon`` is the P-wave anisotropy (the fractional difference between
    horizontal and vertical P velocity), ``delta`` controls the near-vertical
    P moveout and is the one the AVO gradient sees, and ``gamma`` is the S
    anisotropy, which P-P reflectivity does not use.  ``gamma`` is carried
    anyway so a caller can hold a complete description in one object.

    All zero is isotropy, and is the default: this toolkit does not guess a
    rock's anisotropy.
    """

    epsilon: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0

    def __post_init__(self):
        for name in ("epsilon", "delta", "gamma"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}")
            object.__setattr__(self, name, value)

    @property
    def is_isotropic(self):
        return self.epsilon == 0.0 and self.delta == 0.0 and self.gamma == 0.0


#: The zero medium, and what every argument here defaults to.
ISOTROPIC = Thomsen()

#: Published shale values, to size the question rather than to answer it.
#:
#: These are *examples from the literature*, not defaults and not a model of
#: anyone's field.  Thomsen (1986) tabulates dozens of measurements whose
#: delta ranges from slightly negative to about 0.2 in the same lithology, so
#: picking one and calling it "shale" would be a fiction.  Use them to see
#: what size of effect is in play, then get the numbers for your own section.
LITERATURE_SHALES = {
    # Thomsen (1986), Table 1: "Mesaverde (4903) mudshale" and neighbours span
    # this range; the pair below brackets most of the clastic entries.
    "weak": Thomsen(epsilon=0.05, delta=0.02),
    "moderate": Thomsen(epsilon=0.15, delta=0.08),
    "strong": Thomsen(epsilon=0.25, delta=0.15),
}


def _thomsen(value):
    """Accept a :class:`Thomsen`, a mapping, or ``None``."""
    if value is None:
        return ISOTROPIC
    if isinstance(value, Thomsen):
        return value
    if isinstance(value, dict):
        return Thomsen(**{k: value[k] for k in ("epsilon", "delta", "gamma")
                          if k in value})
    raise TypeError("expected a Thomsen, a mapping or None, "
                    f"got {type(value).__name__}")


def ruger_terms(vp1, vs1, rho1, vp2, vs2, rho2, upper=None, lower=None):
    """The three Rüger coefficients ``(A, B, C)`` for one interface.

    Returned separately from the reflectivity because they are the useful
    output: ``B`` is what the class is made of, and having it in closed form
    is what lets a caller say *the anisotropy moved the gradient by this
    much* rather than only *the curve changed*.

    ``R(t) = A + B sin^2(t) + C sin^2(t) tan^2(t)``.

    Parameters
    ----------
    vp1, vs1, rho1 : float
        Upper (incidence) medium.  Velocities are the **vertical** ones,
        which is what a sonic in a vertical hole measures.
    vp2, vs2, rho2 : float
        Lower medium.
    upper, lower : Thomsen, mapping or None
        Anisotropy either side.  ``None`` is isotropic, so omitting both
        recovers the Aki-Richards three-term coefficients exactly.

    Returns
    -------
    tuple of float
        ``(A, B, C)``.  Any non-finite input gives ``(nan, nan, nan)``
        rather than a number that looks computed.
    """
    values = [float(v) for v in (vp1, vs1, rho1, vp2, vs2, rho2)]
    if not all(np.isfinite(values)):
        return (float("nan"),) * 3
    vp1, vs1, rho1, vp2, vs2, rho2 = values

    up, lo = _thomsen(upper), _thomsen(lower)

    alpha = (vp1 + vp2) / 2.0
    beta = (vs1 + vs2) / 2.0
    d_alpha = vp2 - vp1

    # Impedance and shear modulus, as Rüger writes them. Using these rather
    # than the equivalent sum of density and velocity ratios costs nothing:
    # dx/x_bar about an arithmetic mean is a second-order-accurate stand-in
    # for d(ln x), so both forms agree through second order and part company
    # only at third (measured: a ratio of 8 per halving of the contrast, and
    # 7e-5 at ordinary log contrasts). This is the published form, and the one
    # a reader checking against the paper expects to see.
    z1, z2 = rho1 * vp1, rho2 * vp2
    g1, g2 = rho1 * vs1 ** 2, rho2 * vs2 ** 2
    z_bar, g_bar = (z1 + z2) / 2.0, (g1 + g2) / 2.0

    if alpha <= 0 or z_bar <= 0:
        return (float("nan"),) * 3

    # A shear-free medium (a fluid) has no shear-modulus contrast to take a
    # ratio of; the term is zero there rather than a division by zero.
    shear_term = 0.0 if g_bar <= 0 else (2.0 * beta / alpha) ** 2 * (
        (g2 - g1) / g_bar)

    a = 0.5 * (z2 - z1) / z_bar
    b = 0.5 * (d_alpha / alpha - shear_term + (lo.delta - up.delta))
    c = 0.5 * (d_alpha / alpha + (lo.epsilon - up.epsilon))
    return float(a), float(b), float(c)


def ruger_vti_rpp(vp1, vs1, rho1, vp2, vs2, rho2, theta1,
                  upper=None, lower=None):
    """Rüger's VTI P-P reflection coefficient, in degrees.

    Signature matches :func:`avo_qi.core.reflectivity.aki_richards_rpp` with
    two extra keyword arguments, so it drops into the same call sites.

    With ``upper`` and ``lower`` left isotropic this is the Aki-Richards
    three-term response to within the *third*-order difference between
    ``dZ/Z`` and ``da/a + drho/rho`` — under 1e-4 at ordinary log contrasts.
    The anisotropic terms themselves are exact within the linearisation:
    ``delta`` enters the three-term gradient as ``d(delta)/2`` and nothing
    else.  What a two-term Shuey fit recovers is a different question, and
    :func:`fitted_gradient_shift` answers it.

    Notes
    -----
    Weak anisotropy and weak contrast are both assumed, and there is no
    critical-angle behaviour here at all — a linearised form has none. Past
    about 35–40 degrees, or across a large contrast, this is an extrapolation.
    """
    theta = np.radians(np.asarray(theta1, dtype=float))
    a, b, c = ruger_terms(vp1, vs1, rho1, vp2, vs2, rho2,
                          upper=upper, lower=lower)
    sin2 = np.sin(theta) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        tan2 = np.tan(theta) ** 2
    tan2 = np.where(np.isfinite(tan2), tan2, 0.0)
    return a + b * sin2 + c * sin2 * tan2


def gradient_shift(upper=None, lower=None):
    """How far the anisotropy moves the *three-term* gradient: ``d(delta)/2``.

    Exact within the linearisation, and the clean statement of where the
    anisotropy goes: in ``R = A + B sin^2 + C sin^2 tan^2``, delta lands in B
    and nothing else.

    It is **not** what a two-term Shuey fit will see.  Over any real angle
    range ``sin^2 tan^2`` is far from orthogonal to ``sin^2``, so the epsilon
    contrast sitting in C leaks into the fitted gradient as well — on a
    0–40 degree gather with a moderate shale that leak is about the same size
    as the delta term itself.  Use :func:`fitted_gradient_shift` for the
    number the classification actually uses.

    Note the sign: the contrast is lower minus upper, so an isotropic sand
    beneath a shale with positive delta gives a **negative** shift, pushing a
    reflector towards Class III.
    """
    return 0.5 * (_thomsen(lower).delta - _thomsen(upper).delta)


def fitted_gradient_shift(upper=None, lower=None, angles=None):
    """The gradient shift a two-term Shuey fit over ``angles`` recovers.

    The honest counterpart to :func:`gradient_shift`, and the one to compare
    against ``a_tol``: the class comes from a two-term fit, and that fit
    cannot separate ``sin^2 tan^2`` from ``sin^2`` over a finite angle range.
    So the far-angle term — which carries the epsilon contrast — leaks into
    the gradient, and how much depends on how far out the gather goes.

    Both terms are linear in the Thomsen contrasts and the fit is linear, so
    this is exact rather than an estimate: it is the least-squares gradient of
    the anisotropic part of the response on its own.

    Parameters
    ----------
    upper, lower : Thomsen, mapping or None
        The two media.
    angles : array_like
        Incidence angles in degrees, the same ones the reflectors are fitted
        over.  Fewer than two distinct angles gives NaN — a gradient is not
        defined there.

    Returns
    -------
    float
        The shift in the fitted gradient.  Reduces to ``d(delta)/2`` only in
        the limit of a vanishingly narrow angle range, where the third term
        contributes nothing.
    """
    up, lo = _thomsen(upper), _thomsen(lower)
    theta = np.radians(np.atleast_1d(np.asarray(
        [] if angles is None else angles, dtype=float)))
    theta = theta[np.isfinite(theta)]
    if np.unique(theta).size < 2:
        return float("nan")

    sin2 = np.sin(theta) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        tan2 = np.tan(theta) ** 2
    tan2 = np.where(np.isfinite(tan2), tan2, 0.0)

    # The anisotropic part of R on its own, then the gradient a two-term fit
    # takes off it. Superposition does the rest: the isotropic part of the
    # response is unchanged, so the shift in the fitted B is the fitted B of
    # the difference.
    response = (0.5 * (lo.delta - up.delta) * sin2
                + 0.5 * (lo.epsilon - up.epsilon) * sin2 * tan2)
    design = np.column_stack([np.ones_like(sin2), sin2])
    return float(np.linalg.lstsq(design, response, rcond=None)[0][1])


def thomsen_from_vsh(vsh, shale, sand=None):
    """Interpolate Thomsen parameters between a sand and a shale end member.

    A single-number stand-in for a per-sample measurement nobody has: the
    fabric that makes a shale anisotropic is the clay, so scaling with shale
    volume is the least unreasonable thing to do with one curve.

    It is an interpolation and not a physical mixing law.  Anisotropy is not
    additive — a laminated sand-shale sequence can be more anisotropic than
    either component, which is exactly the case this cannot represent.  Use it
    to carry a measured end-member value through a well, not to predict one.

    Returns
    -------
    Thomsen
        The interpolated medium.  ``vsh`` outside [0, 1] is clipped, since a
        shale volume is a fraction and an extrapolated anisotropy is not
        something this is entitled to invent.
    """
    shale, sand = _thomsen(shale), _thomsen(sand)
    f = float(np.clip(float(vsh), 0.0, 1.0)) if np.isfinite(vsh) else np.nan
    if not np.isfinite(f):
        # An unknown shale volume gives an unknown anisotropy, not an
        # isotropic one: a missing curve must not quietly read as "no fabric".
        return None
    return Thomsen(
        epsilon=sand.epsilon + f * (shale.epsilon - sand.epsilon),
        delta=sand.delta + f * (shale.delta - sand.delta),
        gamma=sand.gamma + f * (shale.gamma - sand.gamma),
    )


def thomsen_logs_from_vsh(vsh, shale, sand=None):
    """:func:`thomsen_from_vsh` down a whole log, as three arrays.

    Returns
    -------
    dict
        ``epsilon``, ``delta`` and ``gamma``, each the length of ``vsh``, with
        NaN wherever the shale volume was not known.
    """
    vsh = np.asarray(vsh, dtype=float)
    shale, sand = _thomsen(shale), _thomsen(sand)
    f = np.clip(vsh, 0.0, 1.0)
    out = {}
    for name in ("epsilon", "delta", "gamma"):
        lo, hi = getattr(sand, name), getattr(shale, name)
        out[name] = np.where(np.isfinite(vsh), lo + f * (hi - lo), np.nan)
    return out
