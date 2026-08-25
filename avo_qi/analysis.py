"""The reflector pipeline, in one place so more than one page can run it.

Everything the AVO Classification page computes *before* its filters lived
inside that page as a script.  That was fine while one well was the only well:
the page ran top to bottom on the active one and nothing else needed the
result.  Comparing wells needs the same computation on a well that is not
active, and copying it would guarantee the two drift apart — the blocking rule,
the amplitude cut and the class tolerance would quietly diverge and nobody
would notice until two pages disagreed about the same reflector.

So the pipeline is a function, and the page calls it like anybody else:

    trace -> events -> lobes -> blocked layers -> A and B -> class

The trace decides *where* the reflectors are; the logs are asked only what the
rock does there.  See ``core/synthetic.trace_events`` and
``core/blocking.lobe_windows`` for the two halves of that.

Streamlit enters only through :func:`avo_qi.ui.time_well` and
:func:`avo_qi.ui.build_wavelet`, both of which are caches around pure code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from avo_qi.core.avo import reflector_avo
from avo_qi.core.avo_attributes import (MUDROCK_SLOPE, fluid_factor,
                                        pseudo_shear_reflectivity)
from avo_qi.core.blocking import (blocked_reflectivity, half_cycle_samples,
                                  lobe_windows)
from avo_qi.core.lithology import interface_lithology, lobe_lithology
from avo_qi.core.properties import (lobe_property, net_to_gross, pick_side,
                                    reservoir_side)
from avo_qi.core.zones import zone_of_interface, zone_of_lobe
from avo_qi.core.reflectivity import reflectivity_series
from avo_qi.core.synthetic import build_gather, full_stack, trace_events
from avo_qi.core.tuning import apparent_period, tuning_thickness_from_wavelet

__all__ = ["reflector_analysis", "rc_at"]


def rc_at(reference, samples, values):
    """A full-length RC matrix carrying ``values`` at ``samples``, else zero."""
    out = np.zeros_like(reference)
    out[np.asarray(samples, dtype=int), :] = values
    return out


#: Petrophysical curves averaged over each reflector's half-lobes, where the
#: well carries them.  VSH leads because the reservoir side is decided on it.
LOBE_CURVES = ("VSH", "PHI", "SW")


def reflector_analysis(well, settings, case=None, block_method="backus",
                       guard=2, lithology=True, zones=True, properties=True,
                       attributes=True):
    """Pick every reflector on one well's full stack and classify it.

    Parameters
    ----------
    well : WellData
        Any well, not only the active one — which is the point of the
        function.
    settings : Settings
        The shared analysis settings: sample rate, angles, reflectivity
        method, wavelet, amplitude cut and class tolerance.
    case : str, optional
        Which fluid case to analyse; the well's own active case by default.
        A case name from another well means nothing here, so it is resolved
        against *this* well.
    block_method, guard : str, int
        How the layers either side of an event are averaged, and the fallback
        window's guard where a lobe cannot be found.
    attributes : bool
        Add the derived AVO attributes — pseudo-shear reflectivity, the fluid
        factor and the A·B product — to every event.  They are algebra on A
        and B and cost nothing; what they need that a caller cannot easily
        supply is each reflector's own background Vp/Vs, which comes from the
        blocked layers here.
    lithology, zones, properties : bool
        Add the lithology, the zone and the petrophysics either side of each
        event, all three read over the same half-lobes the elastic properties
        were averaged over.  ``settings`` carries the zonation and the net
        cutoffs, so pass the *well's own* settings (see
        :func:`avo_qi.ui.well_settings_for`) or a well will be zoned by
        another well's tops.

    Returns
    -------
    dict
        The frames and arrays the pages work from — ``table`` (one row per
        event), ``blocked`` and ``lobe_bounds`` (parallel to it *before* any
        filtering), and the trace, gather and wavelet they came from.

    Raises
    ------
    ValueError
        If the well has no samples with Vp, Vs and RHOB together, or fewer
        than two angles are configured. Both are conditions a caller has to
        show rather than paper over.
    """
    from avo_qi.ui import (build_wavelet, lithology_labels, time_well,
                           zone_labels)

    angles = settings.angles
    if angles.size < 2:
        raise ValueError(
            "At least two angles are needed for an intercept-gradient fit.")

    case = case if case in (well.cases or [case]) else well.active_case
    tw = time_well(well, settings, case)
    vp = tw["VP"].to_numpy(float)
    vs = tw["VS"].to_numpy(float)
    rho = tw["RHOB"].to_numpy(float)
    twt = tw["TWT"].to_numpy(float)
    depth = tw["DEPTH"].to_numpy(float) if "DEPTH" in tw.columns else None

    rc = reflectivity_series(vp, vs, rho, angles, method=settings.method)
    _, wavelet = build_wavelet(settings)
    window = half_cycle_samples(apparent_period(wavelet, settings.dt), settings.dt)

    # The gather is built once: the full stack is what the lobes are read from,
    # so blocking needs it before anything else.
    gather = build_gather(vp, vs, rho, angles, wavelet, dt=settings.dt,
                          method=settings.method)
    stack = full_stack(gather)

    out = {
        "well": well, "case": case, "tw": tw, "vp": vp, "vs": vs, "rho": rho,
        "twt": twt, "depth": depth, "angles": angles, "rc": rc,
        "wavelet": wavelet, "window": window, "gather": gather,
        "full_stack": stack,
        "tuning_twt": tuning_thickness_from_wavelet(wavelet, settings.dt),
        "table": pd.DataFrame(), "blocked": None, "lobe_bounds": None,
        "fixed_table": None,
    }

    # Every turning point on the full stack above the amplitude cut is an
    # event. Two consequences, both intended: each event has a lobe of its own
    # by construction, and a thin bed whose top and base interfere into one
    # trough gives one event rather than two — which is what the seismic shows.
    events = trace_events(stack, relative=settings.threshold)
    samples = events["index"]
    out["events"] = events
    out["samples"] = samples
    if not samples.size:
        return out

    fixed = blocked_reflectivity(
        vp, vs, rho, samples, angles, window=window, method=block_method,
        guard=int(guard), reflectivity_method=settings.method)
    fixed_table = reflector_avo(
        rc_at(rc, samples, fixed["rc"]), vp, vs, rho, angles,
        method=settings.method, both=False, depth=depth, twt=twt,
        samples=samples, a_tol=settings.a_tol, mask_post_critical=False)

    # Polarity comes from the trace, not from a log-derived R0: the event is a
    # trough or a peak because that is what the trace does there.
    bounds = lobe_windows(stack, samples,
                          polarity=events["polarity"].astype(float),
                          max_half_width=2 * window)
    blocked = blocked_reflectivity(
        vp, vs, rho, samples, angles, window=window, method=block_method,
        guard=int(guard), reflectivity_method=settings.method, bounds=bounds)

    table = reflector_avo(
        rc_at(rc, samples, blocked["rc"]), vp, vs, rho, angles,
        method=settings.method, both=True, depth=depth, twt=twt,
        samples=samples, a_tol=settings.a_tol, mask_post_critical=False)
    table["critical_angle"] = blocked["critical_angle"]
    table["n_angles"] = (~np.isnan(blocked["rc"])).sum(axis=1)
    table["blocking"] = np.where(blocked["from_lobe"], "lobe", "fixed window")
    table["amplitude"] = events["amplitude"]
    table["polarity"] = np.where(events["polarity"] > 0, "peak", "trough")
    table["lobe_samples"] = (np.asarray(blocked["n_upper"])
                             + np.asarray(blocked["n_lower"]))

    # True vertical depth per event, where the well has a datum. MD is hole
    # length and is not comparable between wells; a class-against-depth trend
    # needs TVDSS and a compaction trend needs TVDBML.
    for reference in ("TVD", "TVDSS", "TVDBML"):
        if reference in tw.columns:
            table[reference.lower()] = tw[reference].to_numpy(float)[samples]
    if "depth" in table.columns:
        lead = [c for c in ("sample", "depth", "tvd", "tvdss", "tvdbml", "twt")
                if c in table.columns]
        table = table[lead + [c for c in table.columns if c not in lead]]

    table["A_fixed"] = fixed_table["A_shuey"].to_numpy()
    table["class_fixed"] = fixed_table["avo_class"].to_numpy()
    table["dA_blocking"] = table["A_shuey"] - table["A_fixed"]
    # `blocked` and `lobe_bounds` are parallel to the table *as blocked*, and
    # the page's filters shorten it and reset the index. Without a row of its
    # own to point back with, a detail panel would quote — and the class
    # probabilities would be computed on — whichever reflector happened to land
    # at that position after filtering, which is a different interface.
    table["props_row"] = np.arange(len(table))

    if lithology:
        # Over the same half-lobes the elastic properties were averaged over,
        # so "shale over sand" names the rock the intercept and gradient
        # actually came from.
        labels = lithology_labels(tw, settings)
        pairs = (lobe_lithology(labels, bounds, samples=samples)
                 if bounds is not None
                 else interface_lithology(labels, samples))
        table = table.assign(litho_upper=pairs["upper"],
                             litho_lower=pairs["lower"],
                             litho_pair=pairs["pair"])
        out["litho"] = labels

    if zones:
        # A top falling anywhere inside an event's lobe is a top that event is
        # carrying; comparing only the two samples at the extremum went quiet
        # once reflectors were picked off the trace.
        labels = zone_labels(tw, well, settings)
        zoning = (zone_of_lobe(labels, bounds, samples=samples)
                  if bounds is not None
                  else zone_of_interface(labels, samples))
        table = table.assign(zone=zoning["zone"], zone_below=zoning["zone_below"],
                             is_zone_boundary=zoning["is_zone_boundary"])
        out["zone_labels"] = labels

    if properties:
        table = table.assign(**_lobe_properties(tw, settings, bounds, samples))

    if attributes:
        table = table.assign(**_avo_attributes(table, blocked, settings))

    out.update({"table": table, "blocked": blocked, "lobe_bounds": bounds,
                "fixed_table": fixed_table})
    return out


def _lobe_properties(tw, settings, bounds, samples):
    """Petrophysics per reflector, over the layers AVO was actually fitted to.

    The point of averaging over the half-lobes rather than reading the two
    samples at the boundary: the class came from a lobe of rock, so the
    porosity it is set against has to come from the same lobe or the two are
    describing different things.

    Every column is suffixed ``_above`` / ``_below``, plus ``d_`` for the
    change downwards across the reflector.  ``*_res`` is whichever side
    :func:`avo_qi.core.properties.reservoir_side` calls the reservoir — a base
    reflector carries its sand *above*, and plotting the class against "the
    porosity below" would set half the reflectors against their seal.
    """
    if bounds is None:
        return {}

    columns = {}
    sides = None
    for curve in LOBE_CURVES:
        if curve not in tw.columns:
            continue
        name = curve.lower()
        # The median, not the mean: PHI and SW carry spikes that a mean over a
        # handful of samples follows straight off the scale.
        found = lobe_property(tw[curve].to_numpy(float), bounds,
                              samples=samples, statistic="median")
        columns[f"{name}_above"] = found["upper"]
        columns[f"{name}_below"] = found["lower"]
        columns[f"d_{name}"] = found["contrast"]
        if curve == "VSH":
            sides = reservoir_side(found["upper"], found["lower"])

    if "VSH" in tw.columns:
        cuts = dict(settings.net_cutoffs)
        pay = bool(settings.net_pay)
        found = net_to_gross(
            tw["VSH"].to_numpy(float), bounds, samples=samples,
            phi=tw["PHI"].to_numpy(float) if pay and "PHI" in tw.columns else None,
            sw=tw["SW"].to_numpy(float) if pay and "SW" in tw.columns else None,
            cutoffs=cuts)
        columns["ntg_above"] = found["upper"]
        columns["ntg_below"] = found["lower"]
        columns["d_ntg"] = found["contrast"]

    if sides is not None:
        columns["reservoir_side"] = sides
        for base in [c[:-6] for c in list(columns) if c.endswith("_above")]:
            columns[f"{base}_res"] = pick_side(sides, columns[f"{base}_above"],
                                               columns[f"{base}_below"])
    return columns


def _avo_attributes(table, blocked, settings):
    """Pseudo-Rs, the fluid factor and the A·B product, per reflector.

    The background Vp/Vs is each reflector's **own upper blocked layer**, not
    a constant 2.0.  It matters: away from 2 the density terms in the
    Aki-Richards form stop cancelling, so a fixed ratio quietly turns a
    density contrast into shear reflectivity.  The upper layer is the right
    side to take it from — the incident medium is what the incidence angle,
    and so the gradient, is measured in.

    Where a lobe gave no usable velocities the ratio falls back to the
    well-wide median rather than to a textbook constant, so the number still
    comes from this well.
    """
    A = table["A_shuey"].to_numpy(float)
    B = table["B_shuey"].to_numpy(float)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.asarray(blocked["vp_upper"], float) / np.asarray(
            blocked["vs_upper"], float)
    ratio = np.where(np.isfinite(ratio) & (ratio > 0), ratio, np.nan)
    fallback = float(np.nanmedian(ratio)) if np.isfinite(ratio).any() else 2.0
    ratio = np.where(np.isfinite(ratio), ratio, fallback)

    slope = float(getattr(settings, "mudrock_slope", MUDROCK_SLOPE))
    rs = np.array([pseudo_shear_reflectivity(a, b, vp_vs=r)
                   for a, b, r in zip(A, B, ratio)], dtype=float)
    factor = np.array([fluid_factor(a, s_, vp_vs=r, slope=slope)
                       for a, s_, r in zip(A, rs, ratio)], dtype=float)
    return {"rp": A, "rs": rs, "fluid_factor": factor, "ab_product": A * B,
            "vp_vs_background": ratio}
