"""Tuned versus untuned AVO: interface response against measured response.

The **untuned** response is the reflection coefficient at a single boundary,
``R(theta)`` from Zoeppritz or Aki-Richards, with no wavelet involved.  It is
what ``core.avo.reflector_avo`` fits.

The **tuned** response is what a seismic amplitude pick actually returns.  Once
the reflectivity is convolved with a wavelet, a bed thinner than a quarter
wavelength has its top and base wavelets overlapping, so the picked extremum is
an interference composite rather than the interface coefficient.  Because the
top and base coefficients vary differently with angle, tuning is itself
angle-dependent: a tuned gradient can differ from the untuned one, and a bed
can change AVO class on thickness alone.

Nothing here is an approximation of the other — they answer different
questions.  Untuned tells you what the rock does; tuned tells you what the
seismic will show.
"""

from __future__ import annotations

import numpy as np

from .reflectivity import reflectivity_series
from .synthetic import build_gather, trace_extrema

__all__ = [
    "wavelet_scale",
    "apparent_period",
    "apparent_frequency",
    "tuning_thickness_from_wavelet",
    "tuning_thickness_twt",
    "tuning_thickness_depth",
    "tuned_amplitudes",
    "tuned_avo",
    "wedge_model",
]


def wavelet_scale(wavelet):
    """Signed extremum of the wavelet — what an isolated reflector is scaled by.

    Convolving a lone spike of amplitude ``R`` with wavelet ``w`` gives ``R w``
    shifted, whose extremum is ``R`` times this value.  It is 1.0 for a
    unit-peak zero-phase wavelet, which is what ``core.wavelet.ricker``
    returns, and it is what makes the pure reflection-coefficient definition
    of "untuned" agree with the thick-bed one.  An uploaded wavelet that is
    not unit peak will not agree, so the two are reported separately.
    """
    wavelet = np.asarray(wavelet, dtype=float).ravel()
    if wavelet.size == 0:
        return 1.0
    return float(wavelet[int(np.argmax(np.abs(wavelet)))])


def apparent_period(wavelet, dt):
    """Apparent period of a wavelet, measured between its two side lobes.

    This is the period that governs tuning, and it is **not** the reciprocal
    of the spectral peak frequency.  A Ricker of peak frequency ``f`` has an
    apparent period of ``sqrt(6) / (pi f)``, so its apparent frequency is
    ``1.28 f`` — using the spectral peak instead overstates the tuning
    thickness by that factor.
    """
    from .synthetic import _local_extrema

    wavelet = np.asarray(wavelet, dtype=float).ravel()
    dt = float(dt)
    turns = _local_extrema(wavelet)
    centre = int(np.argmax(np.abs(wavelet)))
    before = turns[turns < centre]
    after = turns[turns > centre]
    if before.size == 0 or after.size == 0:
        raise ValueError("wavelet has no side lobes to measure a period from")
    return float(after[0] - before[-1]) * dt


def apparent_frequency(wavelet, dt):
    """Apparent (waveform) frequency of a wavelet in Hz — 1 / apparent period."""
    period = apparent_period(wavelet, dt)
    return float("inf") if period == 0 else 1.0 / period


def tuning_thickness_from_wavelet(wavelet, dt):
    """Two-way time thickness at which a bed tunes, measured from the wavelet.

    Half the apparent period.  Prefer this over :func:`tuning_thickness_twt`
    unless you already know the apparent frequency.
    """
    return apparent_period(wavelet, dt) / 2.0


def tuning_thickness_twt(f_apparent):
    """Two-way time thickness at which a bed tunes, in seconds.

    A bed tunes at a quarter wavelength, which in two-way time is
    ``1 / (2 f)`` — independent of velocity, because the velocity cancels.

    ``f_apparent`` must be the **apparent** (waveform) frequency, not the
    spectral peak.  For a Ricker the two differ by a factor of 1.28; pass
    :func:`apparent_frequency`, or use
    :func:`tuning_thickness_from_wavelet` and skip the step.
    """
    f_apparent = float(f_apparent)
    if f_apparent <= 0:
        raise ValueError("apparent frequency must be positive")
    return 1.0 / (2.0 * f_apparent)


def tuning_thickness_depth(f_apparent, velocity):
    """Bed thickness at which it tunes, in metres: a quarter wavelength.

    ``f_apparent`` is the apparent frequency, as for
    :func:`tuning_thickness_twt`.
    """
    f_apparent = float(f_apparent)
    if f_apparent <= 0:
        raise ValueError("apparent frequency must be positive")
    return float(velocity) / (4.0 * f_apparent)


def tuned_amplitudes(gather, samples, half_window=5, polarity=None):
    """Pick each reflector's amplitude on every angle trace of a gather.

    The event is tracked independently per angle — the same way an
    interpreter picks an amplitude across a gather — so the result carries the
    interference between neighbouring reflectors as well as the interface
    response.

    Parameters
    ----------
    gather : ndarray
        ``(n_samples, n_angles)`` from :func:`core.synthetic.build_gather`.
    samples : array_like
        Interface indices to pick.
    half_window : int
        Search radius, in samples, around each interface.
    polarity : array_like, optional
        Expected sign per reflector, normally ``sign(R0)``; keeps a pick from
        jumping onto a neighbouring lobe of the opposite sign.

    Returns
    -------
    dict
        ``amplitude`` and ``index``, both ``(n_reflectors, n_angles)``, and
        ``is_extremum`` flagging picks with no turning point of the right sign.
    """
    gather = np.atleast_2d(np.asarray(gather, dtype=float))
    samples = np.atleast_1d(np.asarray(samples, dtype=int))
    n_angles = gather.shape[1]

    amplitude = np.empty((samples.size, n_angles), dtype=float)
    index = np.empty((samples.size, n_angles), dtype=int)
    is_extremum = np.zeros((samples.size, n_angles), dtype=bool)

    for j in range(n_angles):
        found = trace_extrema(gather[:, j], samples, half_window=half_window,
                              polarity=polarity)
        amplitude[:, j] = found["amplitude"]
        index[:, j] = found["index"]
        is_extremum[:, j] = found["is_extremum"]

    return {"amplitude": amplitude, "index": index, "is_extremum": is_extremum}


def tuned_avo(vp, vs, rho, angles, wavelet, dt=0.001, method="zoeppritz",
              samples=None, threshold=0.01, half_window=5, a_tol=0.02,
              mask_post_critical=True, depth=None, twt=None):
    """Fit intercept and gradient to **tuned** amplitudes, beside the untuned fit.

    Builds the gather, tracks each reflector's amplitude across angle, and fits
    ``A + B sin^2(theta)`` to those picked amplitudes.  The untuned fit from
    :func:`core.avo.reflector_avo` is returned alongside, with the difference
    and any class change that tuning causes.

    Returns
    -------
    pandas.DataFrame
        The untuned table (``A_shuey``/``B_shuey``, in reflection-coefficient
        units) plus ``A_thick``/``B_thick`` — the same interface isolated but
        carried through the wavelet, so in seismic units — and the measured
        ``A_tuned``/``B_tuned``, ``class_tuned``, the difference from the
        thick-bed reference (``dA_tuning``, ``dB_tuning``), ``tuning_ratio``
        and ``tuning_changes_class``.
    """
    from .avo import classify_array, reflector_avo, shuey_fit

    angles = np.atleast_1d(np.asarray(angles, dtype=float))
    rc = reflectivity_series(vp, vs, rho, angles, method=method)

    untuned = reflector_avo(
        rc, vp, vs, rho, angles, method=method, samples=samples,
        threshold=threshold, a_tol=a_tol, mask_post_critical=mask_post_critical,
        depth=depth, twt=twt,
    )
    if untuned.empty:
        return untuned.assign(A_thick=[], B_thick=[], A_tuned=[], B_tuned=[],
                              class_tuned=[], dA_tuning=[], dB_tuning=[],
                              tuning_ratio=[], tuning_changes_class=[])

    gather = build_gather(vp, vs, rho, angles, wavelet, dt=dt, method=method)
    idx = untuned["sample"].to_numpy()
    picks = tuned_amplitudes(gather, idx, half_window=half_window,
                             polarity=np.sign(untuned["R0"].to_numpy(float)))

    measured = picks["amplitude"].copy()
    if mask_post_critical and "critical_angle" in untuned.columns:
        theta_c = untuned["critical_angle"].to_numpy(float)
        for k in range(measured.shape[0]):
            if np.isfinite(theta_c[k]):
                measured[k, angles >= theta_c[k]] = np.nan

    A_t, B_t = shuey_fit(measured, angles)

    # The thick-bed reference: the same interface with nothing near enough to
    # interfere, carried through the same wavelet.  Comparing tuned against
    # this isolates interference from wavelet scaling, so the ratio is 1.0 for
    # a thick bed whatever the wavelet's peak amplitude happens to be.
    scale = wavelet_scale(wavelet)
    A_thick = untuned["A_shuey"].to_numpy(float) * scale
    B_thick = untuned["B_shuey"].to_numpy(float) * scale
    reference = untuned["R0"].to_numpy(float) * scale

    zero_offset = measured[:, int(np.argmin(np.abs(angles)))]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = zero_offset / reference
    ratio = np.where(np.isfinite(ratio), ratio, np.nan)

    class_tuned = classify_array(A_t, B_t, a_tol=a_tol)
    out = untuned.copy()
    out["A_thick"] = A_thick
    out["B_thick"] = B_thick
    out["A_tuned"] = A_t
    out["B_tuned"] = B_t
    out["class_tuned"] = class_tuned
    out["dA_tuning"] = A_t - A_thick
    out["dB_tuning"] = B_t - B_thick
    out["tuning_ratio"] = ratio
    out["tuning_changes_class"] = class_tuned != untuned["avo_class"].to_numpy()
    out.attrs["tuned_amplitudes"] = picks["amplitude"]
    out.attrs["angles"] = angles
    out.attrs["wavelet_scale"] = scale
    return out


def wedge_model(upper, reservoir, lower, thicknesses, angles, wavelet,
                dt=0.001, method="zoeppritz", pad=80, half_window=None):
    """Build a wedge and measure how tuning distorts its AVO response.

    The reservoir is inserted between ``upper`` and ``lower`` at a range of
    thicknesses, in samples of two-way time.  For each thickness the gather is
    built and the top and base amplitudes are tracked across angle, then fitted
    for intercept and gradient.  Comparing those against the thick-bed
    (untuned) values is the classic tuning curve.

    Parameters
    ----------
    upper, reservoir, lower : tuple
        ``(vp, vs, rho)`` for each layer.
    thicknesses : array_like
        Reservoir thicknesses in samples.  Zero means no reservoir at all.
    angles : array_like
        Incidence angles in degrees.
    wavelet : array_like
        Wavelet amplitudes at ``dt``.
    pad : int
        Samples of encasing layer above and below.

    Returns
    -------
    dict
        ``thickness_samples``, ``thickness_twt``, ``top_amplitude`` and
        ``base_amplitude`` (each ``(n_thicknesses, n_angles)``), the tuned
        ``A_top``/``B_top`` per thickness, the untuned ``A_interface`` and
        ``B_interface``, and ``gathers``.
    """
    from .avo import shuey_fit

    angles = np.atleast_1d(np.asarray(angles, dtype=float))
    thicknesses = np.atleast_1d(np.asarray(thicknesses, dtype=int))
    wavelet = np.asarray(wavelet, dtype=float)
    if half_window is None:
        half_window = max(int(wavelet.size // 8), 2)

    n_top = pad
    n_samples = 2 * pad + int(thicknesses.max()) + 1

    top_amp = np.full((thicknesses.size, angles.size), np.nan)
    base_amp = np.full((thicknesses.size, angles.size), np.nan)
    apparent = np.full((thicknesses.size, angles.size), np.nan)
    gathers = []

    # The isolated-interface response: what the top would show with no base
    # nearby to interfere with it.
    rc_top = reflectivity_series(
        np.array([upper[0], reservoir[0]]), np.array([upper[1], reservoir[1]]),
        np.array([upper[2], reservoir[2]]), angles, method=method,
    )[0]
    A_iface, B_iface = shuey_fit(rc_top, angles)
    scale = wavelet_scale(wavelet)
    polarity = np.sign(rc_top[int(np.argmin(np.abs(angles)))] * scale)

    for k, thickness in enumerate(thicknesses):
        vp = np.full(n_samples, float(upper[0]))
        vs = np.full(n_samples, float(upper[1]))
        rho = np.full(n_samples, float(upper[2]))
        base_start = n_top + int(thickness)
        if thickness > 0:
            vp[n_top:base_start] = reservoir[0]
            vs[n_top:base_start] = reservoir[1]
            rho[n_top:base_start] = reservoir[2]
        vp[base_start:] = lower[0]
        vs[base_start:] = lower[1]
        rho[base_start:] = lower[2]

        gather = build_gather(vp, vs, rho, angles, wavelet, dt=dt, method=method)
        gathers.append(gather)

        if thickness > 0:
            top = tuned_amplitudes(gather, [n_top - 1], half_window=half_window,
                                   polarity=[polarity])
            top_amp[k] = top["amplitude"][0]
            base = tuned_amplitudes(gather, [base_start - 1],
                                    half_window=half_window, polarity=[-polarity])
            base_amp[k] = base["amplitude"][0]
            # What an interpreter would *measure* off the section: the time
            # between the two picked events.  Above tuning it tracks the bed;
            # below it the two lobes have merged and it stops shortening,
            # which is why a thin bed reads thicker than it is.
            apparent[k] = (base["index"][0] - top["index"][0]) * float(dt)

    A_top = np.full(thicknesses.size, np.nan)
    B_top = np.full(thicknesses.size, np.nan)
    valid = np.all(np.isfinite(top_amp), axis=1)
    if valid.any():
        A_top[valid], B_top[valid] = shuey_fit(top_amp[valid], angles)

    return {
        "thickness_samples": thicknesses,
        "thickness_twt": thicknesses * float(dt),
        "angles": angles,
        "top_amplitude": top_amp,
        "base_amplitude": base_amp,
        "apparent_thickness_twt": apparent,
        "A_top": A_top,
        "B_top": B_top,
        "A_interface": float(A_iface),
        "B_interface": float(B_iface),
        "A_thick": float(A_iface) * scale,
        "B_thick": float(B_iface) * scale,
        "wavelet_scale": scale,
        "rc_interface": rc_top,
        "reference_amplitude": rc_top * scale,
        "gathers": np.array(gathers),
    }
