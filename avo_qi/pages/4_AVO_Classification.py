"""Page 3 — intercept–gradient analysis and AVO classification of reflectors."""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from avo_qi.core.blocking import (  # noqa: E402
    block_properties,
    blocked_reflectivity,
    half_cycle_samples,
    lobe_windows,
)
from avo_qi.core.lithology import interface_lithology  # noqa: E402
from avo_qi.core.zones import zone_of_interface  # noqa: E402
from avo_qi.core.tuning import (  # noqa: E402
    apparent_period,
    tuned_amplitudes,
    tuning_thickness_from_wavelet,
)
from avo_qi.core.avo import (  # noqa: E402
    CLASSES,
    background_trend,
    classify_array,
    compare_cases,
    reflector_avo,
    shuey_fit,
)
from avo_qi.core.reflectivity import (  # noqa: E402
    aki_richards_rpp,
    reflectivity_series,
    zoeppritz_rpp,
)
from avo_qi.core.wavelet import dominant_frequency  # noqa: E402
from avo_qi.core.synthetic import (  # noqa: E402
    angle_stack,
    build_gather,
    full_stack,
    trace_extrema,
)
from avo_qi.core.uncertainty import (  # noqa: E402
    class_probabilities,
    class_probabilities_from_layers,
    perturb_logs,
)
from avo_qi.ui import (  # noqa: E402
    CLASS_COLOURS,
    ab_crossplot,
    lithology_labels,
    apply_zone_filter,
    zone_labels,
    build_wavelet,
    case_colour,
    classified_trace_figure,
    fluid_vector_crossplot,
    gather_figure,
    resolve_reflector_pick,
    selected_reflector_index,
    page_setup,
    require_well,
    sidebar,
    time_well,
)

page_setup("AVO Classification", icon=":triangular_ruler:")
settings = sidebar()
well = require_well()

try:
    tw = time_well(well, settings, settings.case)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

vp = tw["VP"].to_numpy(float)
vs = tw["VS"].to_numpy(float)
rho = tw["RHOB"].to_numpy(float)
twt = tw["TWT"].to_numpy(float)
depth = tw["DEPTH"].to_numpy(float) if "DEPTH" in tw.columns else None
angles = settings.angles

if angles.size < 2:
    st.error("At least two angles are needed for an intercept–gradient fit.")
    st.stop()


def _rc_at(reference, samples, values):
    """A full-length RC matrix carrying ``values`` at ``samples``, else zero."""
    out = np.zeros_like(reference)
    out[np.asarray(samples, dtype=int), :] = values
    return out


rc = reflectivity_series(vp, vs, rho, angles, method=settings.method)

# --- how the untuned response is measured ------------------------------------
st.subheader("Untuned response")
st.caption(
    "Layer properties come from each reflector's **own lobe** on the full "
    "stack. For a trough, the upper half — from the zero crossing above down "
    "to the extremum — gives the layer above, and the lower half gives the "
    "layer below; the logs are averaged over each. The window is therefore "
    "measured on the data rather than assumed from the wavelet, so it narrows "
    "where interference squeezes the lobe and opens where the reflector stands "
    "alone."
)
c1, c2 = st.columns(2)
block_method = c1.selectbox("Average", ["backus", "mean"],
                            help="Backus is the correct elastic upscaling; the "
                                 "arithmetic mean is easier to read but is not "
                                 "what a wave does.")
guard = c2.number_input(
    "Guard (samples)", 0, 20, 2, 1,
    help="Only used where a lobe cannot be found and the fixed half-cycle "
         "window stands in — the lobe halves meet at the extremum and need no "
         "gap.")

_, page_wavelet = build_wavelet(settings)
window = half_cycle_samples(apparent_period(page_wavelet, settings.dt), settings.dt)
tuning_twt = tuning_thickness_from_wavelet(page_wavelet, settings.dt)

# The gather is built once here rather than again further down: the full stack
# is what the lobes are read from, so blocking needs it before anything else.
page_gather = build_gather(vp, vs, rho, angles, page_wavelet,
                           dt=settings.dt, method=settings.method)
page_full_stack = full_stack(page_gather)

blocked_props = None
lobe_bounds = None
table = reflector_avo(
    rc, vp, vs, rho, angles, method=settings.method, both=True,
    depth=depth, twt=twt, threshold=settings.threshold, a_tol=settings.a_tol,
)

if not table.empty:
    samples = table["sample"].to_numpy()
    # The adjacent-sample pass is kept only to seed the polarity of each lobe
    # and to measure what the lobe window changed; it is no longer offered as
    # a way to classify.
    fixed = blocked_reflectivity(
        vp, vs, rho, samples, angles, window=window, method=block_method,
        guard=int(guard), reflectivity_method=settings.method,
    )
    fixed_table = reflector_avo(
        _rc_at(rc, samples, fixed["rc"]),
        vp, vs, rho, angles, method=settings.method, both=False,
        depth=depth, twt=twt, samples=samples, a_tol=settings.a_tol,
        mask_post_critical=False,
    )

    dominant = dominant_frequency(page_wavelet, settings.dt) or 30.0
    search = max(int(round(0.25 / dominant / settings.dt)), 2)
    lobe_extrema = trace_extrema(page_full_stack, samples, half_window=search,
                                 polarity=np.sign(table["R0"].to_numpy(float)))
    lobe_bounds = lobe_windows(page_full_stack, lobe_extrema["index"],
                               polarity=np.sign(table["R0"].to_numpy(float)),
                               max_half_width=2 * window,
                               is_extremum=lobe_extrema["is_extremum"])

    blocked = blocked_reflectivity(
        vp, vs, rho, samples, angles, window=window, method=block_method,
        guard=int(guard), reflectivity_method=settings.method,
        bounds=lobe_bounds,
    )
    blocked_mask = np.isnan(blocked["rc"])

    rc_for_fit = _rc_at(rc, samples, blocked["rc"])
    # Kept for the reflector detail panel, which has to show the *same* layers
    # the fit was made on.
    blocked_props = blocked
    table = reflector_avo(
        rc_for_fit, vp, vs, rho, angles, method=settings.method, both=True,
        depth=depth, twt=twt, samples=samples,
        a_tol=settings.a_tol, mask_post_critical=False,
    )
    table["critical_angle"] = blocked["critical_angle"]
    table["n_angles"] = (~blocked_mask).sum(axis=1)
    table["blocking"] = np.where(blocked["from_lobe"], "lobe", "fixed window")
    # Why a reflector fell back: no turning point of its own polarity on the
    # full stack means it is buried in a neighbour's lobe and has none of its
    # own to halve. Carried into the table and the CSV so `fixed window` is
    # explicable rather than mysterious.
    table["own_extremum"] = lobe_extrema["is_extremum"]
    table["lobe_samples"] = (
        np.asarray(blocked["n_upper"]) + np.asarray(blocked["n_lower"]))
    table["A_fixed"] = fixed_table["A_shuey"].to_numpy()
    table["class_fixed"] = fixed_table["avo_class"].to_numpy()
    table["dA_blocking"] = table["A_shuey"] - table["A_fixed"]
    # `blocked_props` and `lobe_bounds` are arrays parallel to the table *as
    # blocked*, but the zone, lithology and interface-pair filters below shorten
    # it and reset the index. Without a row of its own to point back with, the
    # detail panel would quote — and the class probabilities would be computed
    # on — whichever reflector happened to land at that position after
    # filtering, which is a different interface.
    table["props_row"] = np.arange(len(table))

    from_lobe = int(np.count_nonzero(blocked["from_lobe"]))
    moved = int((table["avo_class"] != table["class_fixed"]).sum())
    median_lobe = float(np.median(table["lobe_samples"])) if len(table) else 0.0
    c1, c2, c3 = st.columns(3)
    # No deltas here: these are descriptions, and Streamlit renders a delta
    # with an arrow, which would read as "up is better" on numbers that have
    # no direction.
    c1.metric("Median lobe window", f"{median_lobe:.0f} samples")
    c2.metric("Largest change in A", f"{table['dA_blocking'].abs().max():+.4f}")
    c3.metric("Reflectors that change class", moved)
    st.caption(
        f"The median lobe spans {median_lobe * settings.dt * 1000:.0f} ms across "
        f"both halves, against the ±{window}-sample "
        f"({2 * window * settings.dt * 1000:.0f} ms) fixed half cycle it "
        "replaces. The two right-hand figures compare the lobe window with "
        "that fixed one."
    )

    if from_lobe < len(table):
        buried = int(np.count_nonzero(~lobe_extrema["is_extremum"]))
        why = (
            f"{buried} of them have no turning point of their own polarity on "
            "the full stack at all — they are buried in a neighbour's lobe, so "
            "there is nothing of theirs to halve. "
        ) if buried else ""
        st.warning(
            f"{len(table) - from_lobe} of {len(table)} reflector(s) have no "
            f"resolvable lobe on the full stack. {why}The rest have no zero "
            f"crossing within {2 * window} samples, or a lobe too narrow to "
            f"halve. All of them fall back to the fixed ±{window}-sample half "
            "cycle and are marked `fixed window` in the reflector table.",
            icon=":material/warning:")
    if moved:
        st.info(
            f"{moved} reflector(s) classify differently on their own lobe than "
            "on a fixed half cycle. The lobe is the narrower window, so it "
            "keeps more of the contrast instead of averaging it away.",
            icon=":material/info:")

    shared = len(lobe_extrema["index"]) - len(np.unique(lobe_extrema["index"]))
    if shared:
        st.info(
            f"{shared} reflector(s) share a lobe with a neighbour, so they get "
            "the same blocked layers and the same A and B. That is what the "
            "seismic sees: interfaces inside one lobe are not separable at "
            "this bandwidth, and reporting different answers for them would "
            "be inventing resolution the data does not have.",
            icon=":material/info:")
st.divider()

if table.empty:
    st.warning(
        f"No interface exceeds the |R| threshold of {settings.threshold:.3f}. "
        "Lower it in the sidebar."
    )
    st.stop()

# Lithology either side of each reflector.  For AVO the pair is what matters:
# a "shale over sand" top is a different event from a "sand over shale" base.
litho = lithology_labels(tw, settings)
pairs = interface_lithology(litho, table["sample"].to_numpy())
table = table.assign(litho_upper=pairs["upper"], litho_lower=pairs["lower"],
                     litho_pair=pairs["pair"])

# Zone per reflector. An interface whose two sides are in different zones is
# the zone boundary itself, which is usually the reflector of interest.
zone_per_sample = zone_labels(tw, well, settings)
zoning = zone_of_interface(zone_per_sample, table["sample"].to_numpy())
table = table.assign(zone=zoning["zone"], zone_below=zoning["zone_below"],
                     is_zone_boundary=zoning["is_zone_boundary"])
if settings.zones:
    in_zone = apply_zone_filter(table["zone"].to_numpy(), settings) | \
        apply_zone_filter(table["zone_below"].to_numpy(), settings)
    if not in_zone.all():
        st.caption(
            f"Zone filter is hiding {int((~in_zone).sum())} of {len(table)} "
            "reflectors — a reflector is kept when either side of it is in a "
            "selected zone."
        )
    table = table[in_zone].reset_index(drop=True)
    if table.empty:
        st.warning("No reflector lies in the selected zones. Widen the zone "
                   "filter in the sidebar.")
        st.stop()

boundaries = int(table["is_zone_boundary"].sum())
if boundaries:
    st.caption(f"{boundaries} of {len(table)} reflectors sit on a zone boundary.")

selected_litho = set(settings.lithologies or [])
if selected_litho:
    # Read from the table's own columns, not from `pairs`: an earlier filter
    # may already have shortened the table, and a stale full-length mask would
    # not line up with it.
    keep = np.array([(u in selected_litho) or (lo in selected_litho)
                     for u, lo in zip(table["litho_upper"], table["litho_lower"])],
                    dtype=bool)
    if not keep.all():
        st.caption(
            f"Lithology filter is hiding {int((~keep).sum())} of {len(table)} "
            "reflectors — a reflector is kept when either side of it is a "
            "selected lithology."
        )
    table = table[keep].reset_index(drop=True)
    if table.empty:
        st.warning(
            "No reflector has a selected lithology on either side. Widen the "
            "lithology filter in the sidebar."
        )
        st.stop()

# Interface pairs. The sidebar filter above asks "is either side a sand?"; this
# one asks for the *pair*, which is the question AVO is actually about — a
# shale-over-sand top and a shale-over-shale contrast can sit side by side in
# the same interval, and mixing them is what smears an A-B cloud. Keeping only
# the tops is usually the difference between a readable crossplot and a blob.
pair_options = sorted(set(table["litho_pair"].astype(str)))
if len(pair_options) > 1:
    default_pairs = [p for p in pair_options if "undefined" not in p] or pair_options
    chosen_pairs = st.multiselect(
        "Interface pairs to keep", pair_options, default=default_pairs,
        help="Upper lithology over lower. Leave only 'shale over sand' to see "
             "reservoir tops alone; clear the box to keep every pair.",
    )
    if chosen_pairs:
        keep_pair = table["litho_pair"].astype(str).isin(chosen_pairs).to_numpy()
        if not keep_pair.all():
            st.caption(
                f"Interface-pair filter is hiding {int((~keep_pair).sum())} of "
                f"{len(table)} reflectors — this filter is on the pair, so "
                "'shale over sand' keeps reservoir tops while dropping their "
                "bases and any shale-over-shale contrast."
            )
        table = table[keep_pair].reset_index(drop=True)
        if table.empty:
            st.warning("No reflector has a selected interface pair. Widen the "
                       "selection above.")
            st.stop()

# Separability. A class is computed from the *logs* — the Zoeppritz response
# between the two blocked layers — so every interface gets one whether or not
# the seismic can see it. `own_extremum` is the other question: does this
# reflector produce a turning point of its own polarity on the full stack? Where
# it does not, the class is a statement about the interface, not about anything
# you could pick: the amplitude at that time belongs to a neighbour. Both are
# worth having, so they are separated rather than merged.
if "own_extremum" in table.columns:
    buried_here = int((~table["own_extremum"]).sum())
    if buried_here:
        separable_only = st.checkbox(
            f"Only reflectors the seismic can separate "
            f"({len(table) - buried_here} of {len(table)})",
            value=False,
            help="Hides reflectors with no turning point of their own polarity "
                 "on the full stack. Their class is still real interface "
                 "physics, but it is modelled rather than observable — you "
                 "could not pick that event, because the amplitude there "
                 "belongs to a neighbouring reflector.",
        )
        st.caption(
            f"{buried_here} of {len(table)} reflectors "
            f"({buried_here / len(table):.0%}) are buried in a neighbour's lobe. "
            "They keep a class — the interface is real — but it is a modelled "
            "answer, not a measurable one, and they are blocked on the fixed "
            "half cycle rather than on a lobe of their own. They also pull the "
            "background trend and the class counts, so it is worth seeing the "
            "crossplot both ways."
        )
        if separable_only:
            table = table[table["own_extremum"].to_numpy(bool)].reset_index(drop=True)
            if table.empty:
                st.warning("No reflector has an extremum of its own on the full "
                           "stack. Untick the box above.")
                st.stop()

trend = background_trend(table["A_shuey"], table["B_shuey"])

# ------------------------------------------------------------- summary -----
counts = table["avo_class"].value_counts()
st.subheader("Reflector summary")
cols = st.columns(len(CLASS_COLOURS))
for col, label in zip(cols, CLASS_COLOURS):
    col.metric(f"Class {label}" if label != "background/other" else "Background",
               int(counts.get(label, 0)))
st.caption(
    f"{len(table)} reflectors above |R| > {settings.threshold:.3f}, "
    f"classified at a_tol = {settings.a_tol:.3f} using "
    f"{settings.method.replace('_', '-')} reflectivity over "
    f"{angles[0]:.0f}–{angles[-1]:.0f}°."
)

known_pairs = [p for p in table["litho_pair"].unique() if "undefined" not in p]
if known_pairs:
    with st.expander(f"Lithology pairs ({len(known_pairs)} distinct)"):
        summary = (
            table.groupby(["litho_pair", "avo_class"]).size()
            .rename("reflectors").reset_index()
            .sort_values("reflectors", ascending=False)
        )
        st.dataframe(summary, use_container_width=True, hide_index=True)
        st.caption(
            "Reservoir tops are the *over sand* pairs; their bases are the "
            "*sand over* ones. Use **Interface pairs to keep** above to leave "
            "only shale over sand — mixing tops, bases and shale-on-shale is "
            "what smears an A-B cloud."
        )

# --------------------------------------------------------- A-B crossplot ---
st.divider()
left, right = st.columns([3, 2])
with left:
    st.subheader("Intercept–gradient crossplot")
    st.plotly_chart(ab_crossplot(table, trend=trend, a_tol=settings.a_tol),
                    use_container_width=True)
with right:
    st.subheader("Background trend")
    if np.isfinite(trend.slope):
        st.latex(rf"B = {trend.slope:.3f}\,A {trend.intercept:+.4f}")
        st.caption(
            f"Robust (Theil-Sen) fit through {trend.n_points} reflectors. Distance from "
            "this line is the fluid-anomaly measure: points far below the trend are "
            "candidate hydrocarbon responses."
        )
        dev = trend.deviation(table["A_shuey"], table["B_shuey"])
        table = table.assign(background_deviation=dev)
        worst = table.iloc[int(np.argmin(dev))]
        label = f"{worst['depth']:.1f} m" if "depth" in table.columns else f"sample {int(worst['sample'])}"
        st.metric("Largest negative deviation", f"{dev.min():+.4f}", label)
    else:
        st.info("Not enough reflectors to fit a background trend.")

# ---------------------------------------------------- class confidence -----
st.divider()
st.subheader("How sure is each class?")
st.caption(
    "A class label reads as a fact. It is really the answer to *where do the "
    "intercept and gradient land*, and both are computed from logs with a "
    "measurement error. Perturbing the logs within that error and reclassifying "
    "each time turns the label into odds — and a reflector sitting near a class "
    "boundary shows up as one, instead of hiding behind a confident name."
)

confidence_on = st.checkbox("Estimate class probabilities", value=False,
                            key="avo_confidence_on")
st.caption(
    "Reclassified on the same **lobe windows** the labels above use, so the "
    "odds and the label describe the same interface."
)
q1, q2, q3, q4 = st.columns(4)
n_draws = q1.slider("Realisations", 50, 1000, 300, 50, disabled=not confidence_on)
pct_vp = q2.slider("Vp ± (%)", 0.0, 10.0, 1.0, 0.25, disabled=not confidence_on)
pct_vs = q3.slider("Vs ± (%)", 0.0, 15.0, 2.0, 0.25, disabled=not confidence_on,
                   help="A shear sonic is the noisier measurement, and Vs is "
                        "what carries the gradient.")
pct_rho = q4.slider("RHOB ± (%)", 0.0, 10.0, 1.0, 0.25, disabled=not confidence_on)

probabilities = None
if confidence_on and len(table):
    reflector_samples = table["sample"].to_numpy()
    # Bounds are per-reflector and were built before any filter, so they have to
    # be cut down to the rows still on screen or `block_properties` would pair
    # each surviving sample with a stranger's window.
    rows = table["props_row"].to_numpy(int)
    lobe_here = {k: np.asarray(v)[rows] for k, v in lobe_bounds.items()}
    with st.spinner(f"Reclassifying {n_draws} realisations…"):
        realisations = perturb_logs(vp, vs, rho, n_realisations=int(n_draws),
                                    seed=0, vp_pct=pct_vp, vs_pct=pct_vs,
                                    rho_pct=pct_rho)
        # The label above is fitted to the lobe windows, so the odds must be
        # too: classify a different interface and every disagreement with the
        # label is that mismatch rather than a finding. The lobe bounds are
        # held fixed across realisations — they come from the full stack of the
        # unperturbed logs, which is the interface being asked about — while
        # re-averaging each realisation carries the noise through the window,
        # where most of it cancels.
        keys = ("vp_upper", "vs_upper", "rho_upper",
                "vp_lower", "vs_lower", "rho_lower")
        stacked = {k: np.empty((int(n_draws), reflector_samples.size))
                   for k in keys}
        for r in range(int(n_draws)):
            one = block_properties(
                realisations["VP"][r], realisations["VS"][r],
                realisations["RHOB"][r], reflector_samples,
                window=window, method=block_method, guard=int(guard),
                bounds=lobe_here)
            for k in keys:
                stacked[k][r] = one[k]
        probabilities = class_probabilities_from_layers(
            (stacked["vp_upper"], stacked["vs_upper"], stacked["rho_upper"]),
            (stacked["vp_lower"], stacked["vs_lower"], stacked["rho_lower"]),
            angles, samples=reflector_samples, a_tol=settings.a_tol)

    merged = table[["sample"]].merge(probabilities, on="sample", how="left")
    agree = (merged["modal_class"].to_numpy() == table["avo_class"].to_numpy())
    ambiguous = merged["confidence"] < 0.5

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Median confidence", f"{merged['confidence'].median():.0%}",
              help="Usually high, and on its own it hides the reflectors that "
                   "matter — a median says nothing about the weakest one.")
    c2.metric("Least confident", f"{merged['confidence'].min():.0%}",
              help="The reflector whose class is least secure. This is the "
                   "number the median is hiding.")
    c3.metric("Ambiguous reflectors", f"{int(ambiguous.sum())} of {len(merged)}",
              help="Where the most likely class holds less than half the "
                   "realisations, no class is really being asserted.")
    c4.metric("Modal class differs from the label", f"{int((~agree).sum())}",
              help="The deterministic label is the single best estimate; the "
                   "modal class is the commonest outcome under noise. They "
                   "part company where the answer sits on a boundary.",
              delta_color="off")

    # An interpreter thinks in depth, not in sample index.
    if "depth" in table.columns and np.isfinite(table["depth"]).any():
        bar_labels = [f"{d:,.0f} m" for d in table["depth"].to_numpy(float)]
        bar_axis = "Reflector depth"
    else:
        bar_labels = merged["sample"].astype(str).tolist()
        bar_axis = "Reflector (sample index)"

    bars = go.Figure()
    for name in CLASSES:
        if name not in merged.columns:
            continue
        bars.add_trace(go.Bar(
            x=bar_labels, y=merged[name], name=name,
            marker_color=CLASS_COLOURS.get(name, "#999"),
            hovertemplate=("%{x}<br>" + name + " %{y:.0%}<extra></extra>")))
    bars.update_layout(
        barmode="stack", height=420, xaxis_title=bar_axis, xaxis_type="category",
        yaxis_title="Probability", yaxis_tickformat=".0%",
        margin=dict(l=60, r=20, t=40, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(bars, use_container_width=True)

    if ambiguous.any():
        worst = merged.loc[ambiguous].sort_values("confidence")
        lines = []
        for _, row in worst.head(5).iterrows():
            odds = sorted(((row[c], c) for c in CLASSES if c in row),
                          reverse=True)[:2]
            lines.append(
                f"sample {int(row['sample'])}: "
                + " vs ".join(f"{name} {p:.0%}" for p, name in odds))
        st.warning(
            "No class is really being asserted at "
            + "; ".join(lines)
            + ". At this log accuracy those reflectors could go either way.",
            icon=":material/warning:")
    else:
        st.success(
            "Every reflector holds its class in at least half the realisations "
            "at this log accuracy.", icon=":material/check_circle:")

    if (~agree).any():
        disagreeing = [
            f"sample {int(merged['sample'].iloc[i])} labelled "
            f"{table['avo_class'].iloc[i]} but most often "
            f"{merged['modal_class'].iloc[i]}"
            for i in np.flatnonzero(~agree)[:5]
        ]
        st.info(
            "The single best estimate is not the commonest outcome at "
            + "; ".join(disagreeing)
            + ". Both readings are defensible — the reflector is near a class "
              "boundary, which is the thing worth knowing.",
            icon=":material/info:")
elif confidence_on:
    st.info("No reflectors to classify.", icon=":material/info:")

# --------------------------------------------------------------- table -----
st.divider()
st.subheader("Reflector table")

show = table.drop(columns=["props_row"], errors="ignore").copy()
if probabilities is not None:
    # Carried into the table and the CSV, so the odds travel with the
    # label rather than living only on screen.
    keep = ["sample", "modal_class", "confidence"] + [
        c for c in CLASSES if c in probabilities.columns]
    show = show.merge(probabilities[keep], on="sample", how="left")
    for col in ["confidence"] + [c for c in CLASSES if c in show.columns]:
        show[col] = show[col].astype(float).round(3)
if "twt" in show.columns:
    show["twt"] = show["twt"].round(4)
if "depth" in show.columns:
    show["depth"] = show["depth"].round(2)
for col in ("R0", "A_shuey", "B_shuey", "A_ar", "B_ar", "dA", "dB", "background_deviation"):
    if col in show.columns:
        show[col] = show[col].round(5)
if "critical_angle" in show.columns:
    show["critical_angle"] = show["critical_angle"].round(1)

class_filter = st.multiselect("Filter by class", list(CLASS_COLOURS),
                              default=list(CLASS_COLOURS))
filtered = show[show["avo_class"].isin(class_filter)].reset_index(drop=True)
st.caption("Click a row to inspect that reflector in the detail section below.")
row_event = st.dataframe(
    filtered, use_container_width=True, height=340,
    on_select="rerun", selection_mode="single-row", key="reflector_rows",
)

# Row selection is Streamlit's own, rather than Plotly's, so it is the
# dependable way in: a click on the trace is a nicety, a click on the row
# always works.
_rows = []
if row_event is not None:
    _selection = getattr(row_event, "selection", None)
    if _selection is None and isinstance(row_event, dict):
        _selection = row_event.get("selection")
    _rows = list(getattr(_selection, "rows", None)
                 or (_selection or {}).get("rows", []) or [])
if _rows and 0 <= _rows[0] < len(filtered):
    _sample = int(filtered.iloc[_rows[0]]["sample"])
    _match = np.flatnonzero(table["sample"].to_numpy() == _sample)
    if _match.size and _sample != st.session_state.get("_last_row_click"):
        st.session_state["_last_row_click"] = _sample
        st.session_state["reflector_pick"] = int(_match[0])

max_delta = float(np.nanmax(np.abs(table[["dA", "dB"]].to_numpy(float)))) if len(table) else 0.0
st.caption(
    f"`dA` and `dB` are Shuey minus Aki-Richards for the same reflector; the largest "
    f"absolute difference here is {max_delta:.2e}."
)
st.download_button("Download reflector table (CSV)", show.to_csv(index=False).encode(),
                   file_name=f"{well.name}_avo_reflectors.csv", mime="text/csv")

# ---------------------------------------------------- reflector detail -----
st.divider()
st.subheader("Reflector detail")
st.caption(
    "Pick a reflector with the dropdown, by clicking a row in the table above, "
    "or by clicking a marker on the trace — the dropdown and "
    "the panel below follow it. Each marker sits on the amplitude extremum the "
    "reflector produces, coloured by its AVO class. The Vp, Vs and RHOB tracks "
    "and the angle gather share the trace's two-way-time axis, so a reflector "
    "lines up with the contrast that made it on one side and with its own "
    "behaviour against angle on the other."
)

# The gather this page's trace is drawn from.
# Built once near the top, where the lobes were read from it.
detail_wavelet = page_wavelet
detail_gather = page_gather

trace_options = ["Full stack", "Near", "Mid", "Far"] + [f"{a:.0f}°" for a in angles]
trace_choice = st.selectbox("Trace", trace_options, index=0)
bands = {"Near": settings.near, "Mid": settings.mid, "Far": settings.far}
try:
    if trace_choice == "Full stack":
        detail_trace = full_stack(detail_gather)
    elif trace_choice in bands:
        detail_trace = angle_stack(detail_gather, bands[trace_choice], angles)
    else:
        detail_trace = detail_gather[:, trace_options.index(trace_choice) - 4]
except ValueError:
    st.warning(f"No angles fall in the {trace_choice.lower()} band; showing the full stack.")
    detail_trace = full_stack(detail_gather)

# Search radius: a quarter of the wavelet's dominant period, which is where a
# zero-phase wavelet puts the extremum belonging to an interface.
dominant = dominant_frequency(detail_wavelet, settings.dt) or 30.0
half_window = max(int(round(0.25 / dominant / settings.dt)), 2)
# Match each interface to a turning point of the same sign as its
# normal-incidence coefficient, so a weak reflector cannot be handed its
# loud neighbour's lobe.
extrema = trace_extrema(detail_trace, table["sample"].to_numpy(),
                        half_window=half_window,
                        polarity=np.sign(table["R0"].to_numpy(float)))
marker_twt = twt[np.clip(extrema["index"], 0, twt.size - 1)]


def _label(row):
    where = f"{row['depth']:.1f} m" if "depth" in table.columns else f"sample {int(row['sample'])}"
    when = f" / {row['twt']:.3f} s" if "twt" in table.columns else ""
    pair = row.get("litho_pair", "")
    lith = f" · {pair}" if pair and "undefined" not in str(pair) else ""
    return (f"{where}{when} — Class {row['avo_class']} "
            f"(A {row['A_shuey']:+.3f}, B {row['B_shuey']:+.3f}){lith}")


labels = [_label(r) for _, r in table.iterrows()]
# Open on the strongest reflector rather than the shallowest, which on a noisy
# log is often a near-zero interface that happens to clear the threshold.
strongest = int(np.argmax(np.abs(table["R0"].to_numpy(float))))

# A click on the trace arrives in this run's session state, before the
# selectbox below is drawn, so reading it here lets the click drive the
# dropdown instead of fighting it.  Only act on a *new* selection, or manual
# changes to the dropdown would be overridden on every rerun.
clicked = selected_reflector_index(st.session_state.get("class_trace"), marker_twt)
if clicked is not None and clicked != st.session_state.get("_last_trace_click"):
    st.session_state["_last_trace_click"] = clicked
    st.session_state["reflector_pick"] = int(clicked)

# What the dropdown left behind is not necessarily a row index — see
# `resolve_reflector_pick`.  The sample recorded below is the anchor that keeps
# the same reflector selected when a filter renumbers the table.
pick_samples = table["sample"].to_numpy()
st.session_state["reflector_pick"] = resolve_reflector_pick(
    st.session_state.get("reflector_pick"), pick_samples,
    anchor_sample=st.session_state.get("reflector_pick_sample"),
    fallback=strongest,
)
st.session_state["reflector_pick_sample"] = int(
    pick_samples[st.session_state["reflector_pick"]])

d1, d2, d3, d4 = st.columns([2, 1, 1, 1])
detail_height = d1.slider(
    "Panel height (px)", 500, 2200, 1100, 50,
    help="The trace spans the whole analysis window; raise this to read a "
         "long well without squinting.")
show_logs = d2.checkbox("Vp, Vs and RHOB tracks", value=True,
                        help="Drawn to the left of the trace on the same "
                             "two-way-time axis, so a reflector lines up with "
                             "the contrast that produced it.")
show_gather = d3.checkbox("Gather", value=True,
                          help="The angle gather beside the stack, on the same "
                               "two-way-time axis: the stack says where the "
                               "reflector is, the gather says how it behaves "
                               "with angle, which is what the class is about.")
show_lobe = d4.checkbox("Lobe halves", value=True,
                        help="Shade the two half-lobes the selected "
                             "reflector's layers were averaged over.")

trace_col, detail_col = st.columns([3, 2])

with trace_col:
    st.markdown(f"**{trace_choice}** — reflectors by class")
    detail_logs = {"Vp (m/s)": vp, "Vs (m/s)": vs, "RHOB (g/cc)": rho} if show_logs else None
    # Cut to the rows still on screen, so the shading follows the same
    # reflector the dropdown and the curve below are describing.
    detail_lobe = None
    if show_lobe and lobe_bounds is not None:
        _lobe_rows = table["props_row"].to_numpy(int)
        detail_lobe = {k: np.asarray(v)[_lobe_rows] for k, v in lobe_bounds.items()}
    st.plotly_chart(
        classified_trace_figure(
            detail_trace, twt, table, extrema,
            selected=st.session_state["reflector_pick"],
            height=int(detail_height), logs=detail_logs,
            trace_title=trace_choice, lobe=detail_lobe,
            gather=detail_gather if show_gather else None,
            gather_angles=angles if show_gather else None,
        ),
        use_container_width=True, key="class_trace",
        on_select="rerun", selection_mode="points",
    )
    if show_lobe:
        st.caption(
            "The blue band is the **upper** half-lobe and the orange band the "
            "**lower** one, for the selected reflector. Those are the samples "
            "averaged into the two layers behind its A and B — shaded across "
            "the logs as well as the trace, so the window can be read against "
            "the contrast it is meant to capture. A reflector that fell back "
            "to the fixed half cycle has no lobe to shade."
        )
    off = extrema["offset"]
    if np.any(np.abs(off) > 0):
        st.caption(
            f"Extrema searched ±{half_window} samples around each interface "
            f"({dominant:.0f} Hz dominant); largest shift {int(np.abs(off).max())} samples."
        )
    interfering = int(np.sum(~extrema["is_extremum"]))
    if interfering:
        st.caption(
            f"{interfering} reflector(s) have no turning point of their own "
            "polarity within ±{} samples — they are buried in a neighbour's "
            "lobe. Those markers sit at the interface time and are drawn "
            "hollow; their amplitude is not an extremum.".format(half_window)
        )

pick = int(st.session_state["reflector_pick"])

with detail_col:
    st.selectbox(
        "Reflector", range(len(labels)), key="reflector_pick",
        format_func=lambda i: labels[i],
        help="Kept in step with the trace — clicking a marker moves this too.",
    )
    pick = int(st.session_state["reflector_pick"])
    row = table.iloc[pick]
    i = int(row["sample"])

    fine = np.linspace(float(angles[0]), float(angles[-1]), 200)
    sin2_fine = np.sin(np.radians(fine)) ** 2

    # The panel must show the layers the fit was actually made on.  With
    # blocking switched on the A and B above come from a half cycle averaged
    # either side of the boundary, and plotting them over the adjacent-sample
    # coefficient compares two different interfaces: on a gradational boundary
    # the adjacent pair carries a fraction of the contrast, so the curves can
    # differ by more than 0.25 in Rpp and the fit looks badly wrong when it is
    # not.
    if blocked_props is not None:
        # Through `props_row`, never through `pick`: the filters above have
        # renumbered the table, and `pick` is a position in the filtered one.
        p = int(row["props_row"])
        layer = (blocked_props["vp_upper"][p], blocked_props["vs_upper"][p],
                 blocked_props["rho_upper"][p], blocked_props["vp_lower"][p],
                 blocked_props["vs_lower"][p], blocked_props["rho_lower"][p])
        modelled = np.asarray(blocked_props["rc"])[p, :]
        n_lobe = int(blocked_props["n_upper"][p] + blocked_props["n_lower"][p])
        layer_source = (
            f"the reflector's own lobe ({n_lobe} samples across both halves)"
            if blocked_props["from_lobe"][p]
            else f"the fallback fixed half cycle (±{window} samples)")
    else:
        layer = (vp[i], vs[i], rho[i], vp[i + 1], vs[i + 1], rho[i + 1]) \
            if i + 1 < vp.size else None
        modelled = rc[i, :]
        layer_source = "adjacent samples"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=angles, y=modelled, mode="markers",
        name=f"{settings.method.replace('_', '-')} (modelled)",
        marker=dict(size=9, color="#333"),
    ))
    fig.add_trace(go.Scatter(
        x=fine, y=row["A_shuey"] + row["B_shuey"] * sin2_fine, mode="lines",
        name=f"Shuey fit (A {row['A_shuey']:+.3f}, B {row['B_shuey']:+.3f})",
        line=dict(width=2.5, color=CLASS_COLOURS.get(row["avo_class"], "#1f77b4")),
    ))
    fig.add_trace(go.Scatter(
        x=fine, y=row["A_ar"] + row["B_ar"] * sin2_fine, mode="lines",
        name=f"Aki-Richards fit (A {row['A_ar']:+.3f}, B {row['B_ar']:+.3f})",
        line=dict(width=2, dash="dash", color="#ff7f0e"),
    ))

    # The two exact interface models, on those same layers.
    if layer is not None and all(np.isfinite(v) for v in layer):
        fig.add_trace(go.Scatter(x=fine, y=zoeppritz_rpp(*layer, fine), mode="lines",
                                 name="Zoeppritz", line=dict(width=1, color="#999")))
        fig.add_trace(go.Scatter(x=fine, y=aki_richards_rpp(*layer, fine), mode="lines",
                                 name="Aki-Richards (3-term)",
                                 line=dict(width=1, dash="dot", color="#999")))

    # With blocking on, the adjacent-sample curve is worth seeing precisely
    # because it differs — that gap is what blocking is for — but it is drawn
    # faintly and named, never mixed in with the fitted layers.
    if blocked_props is not None and i + 1 < vp.size:
        adjacent_layer = (vp[i], vs[i], rho[i], vp[i + 1], vs[i + 1], rho[i + 1])
        fig.add_trace(go.Scatter(
            x=fine, y=zoeppritz_rpp(*adjacent_layer, fine), mode="lines",
            name="Zoeppritz (adjacent samples, not fitted)",
            line=dict(width=1, dash="dash", color="#c9c9c9")))

    # Past the critical angle the exact solution is complex; its real
    # continuation spikes and is excluded from the fit, so mark that region.
    theta_c = float(row.get("critical_angle", np.nan))
    if np.isfinite(theta_c) and theta_c < angles[-1]:
        fig.add_vrect(x0=theta_c, x1=float(angles[-1]), fillcolor="#d62728",
                      opacity=0.08, line_width=0, layer="below")
        fig.add_vline(x=theta_c, line=dict(color="#d62728", width=1.5, dash="dot"),
                      annotation_text=f"critical {theta_c:.0f}°",
                      annotation_position="top left")

    fig.add_hline(y=0, line=dict(color="#bbb", width=1))
    # Scale to what was fitted.  The unfitted adjacent-sample curve runs past
    # its own critical angle, where the real continuation of the exact solution
    # spikes; left to set the axis it squashes the actual data into a strip.
    fitted_values = np.concatenate([
        np.asarray(modelled, dtype=float).ravel(),
        row["A_shuey"] + row["B_shuey"] * sin2_fine,
        row["A_ar"] + row["B_ar"] * sin2_fine,
    ])
    fitted_values = fitted_values[np.isfinite(fitted_values)]
    if fitted_values.size:
        span = float(np.ptp(fitted_values)) or max(abs(float(fitted_values[0])), 0.05)
        low = float(fitted_values.min()) - 0.35 * span
        high = float(fitted_values.max()) + 0.35 * span
        fig.update_yaxes(range=[low, high])

    fig.update_layout(xaxis_title="Incidence angle (deg)", yaxis_title="Rpp",
                      height=460, margin=dict(l=60, r=20, t=30, b=45),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Modelled on **{layer_source}** — the same layers the A and B above "
        "were fitted to."
        + (" The faint dashed curve is the adjacent-sample coefficient, which "
           "is not what was fitted; where it sits well away from the others "
           "the boundary is gradational and the adjacent pair carries only "
           "part of the contrast." if blocked_props is not None else "")
    )

    if np.isfinite(theta_c) and theta_c < angles[-1]:
        st.warning(
            f"Critical angle at **{theta_c:.1f}°** — angles beyond it are excluded "
            f"from the fit ({int(row['n_angles'])} of {angles.size} angles used). "
            "The shaded amplitudes are the real continuation of a solution that is "
            "genuinely complex there; fitting them would drag the gradient positive "
            "and misclassify the event.",
            icon=":material/warning:",
        )

    c1, c2, c3, c4, c5 = st.columns(5)
    # "background/other" overflows a narrow metric column.
    c1.metric("Class", "backgrnd" if row["avo_class"] == "background/other"
              else row["avo_class"])
    c2.metric("Intercept A", f"{row['A_shuey']:+.4f}")
    c3.metric("Gradient B", f"{row['B_shuey']:+.4f}")
    c4.metric("ΔA Shuey−AkiR", f"{row['dA']:+.1e}")
    c5.metric("ΔB Shuey−AkiR", f"{row['dB']:+.1e}")

    pair = row.get("litho_pair", "")
    if pair and "undefined" not in str(pair):
        st.caption(f"Lithology: **{pair}**.")
    # The properties quoted are the ones behind A and B, so they follow the
    # same blocking choice as the curve above rather than always naming the
    # two samples either side of the boundary.
    if layer is not None and all(np.isfinite(v) for v in layer):
        up_vp, up_vs, up_rho, lo_vp, lo_vs, lo_rho = (float(v) for v in layer)
        st.caption(
            f"Upper layer Vp {up_vp:.0f} m/s, Vs {up_vs:.0f} m/s, ρ {up_rho:.3f} g/cc "
            f"(Vp/Vs {up_vp / up_vs:.2f}) over lower layer Vp {lo_vp:.0f} m/s, "
            f"Vs {lo_vs:.0f} m/s, ρ {lo_rho:.3f} g/cc "
            f"(Vp/Vs {lo_vp / lo_vs:.2f}) — from {layer_source}."
        )

# ---------------------------------------------------------------- tuning ---
st.divider()
st.header("Tuned vs untuned")
st.caption(
    "**Untuned** is the interface response — the reflection coefficient at the "
    "boundary, with no wavelet. **Tuned** is what a seismic pick returns: once "
    "the wavelet is convolved, a bed thinner than a quarter wavelength has its "
    "top and base overlapping, so the picked amplitude is an interference "
    "composite. Because the top and base vary differently with angle, tuning "
    "moves the gradient too — a bed can change AVO class on thickness alone."
)

tuned_gather = build_gather(vp, vs, rho, angles, page_wavelet, dt=settings.dt,
                            method=settings.method)
picks = tuned_amplitudes(
    tuned_gather, table["sample"].to_numpy(),
    half_window=max(window // 2, 2),
    polarity=np.sign(table["R0"].to_numpy(float)),
)
measured = picks["amplitude"].copy()
theta_c = table["critical_angle"].to_numpy(float)
for k in range(measured.shape[0]):
    if np.isfinite(theta_c[k]):
        measured[k, angles >= theta_c[k]] = np.nan

scale = float(page_wavelet[int(np.argmax(np.abs(page_wavelet)))])
A_tuned, B_tuned = shuey_fit(measured, angles)
A_thick = table["A_shuey"].to_numpy(float) * scale
B_thick = table["B_shuey"].to_numpy(float) * scale
class_tuned = classify_array(A_tuned, B_tuned, a_tol=settings.a_tol)
changed = class_tuned != table["avo_class"].to_numpy()

c1, c2, c3 = st.columns(3)
c1.metric("Tuning thickness", f"{tuning_twt * 1000:.1f} ms TWT",
          "half a wavelet cycle")
c2.metric("Largest gradient shift", f"{np.nanmax(np.abs(B_tuned - B_thick)):+.4f}")
c3.metric("Reflectors that change class", int(changed.sum()),
          delta=None if not changed.any() else "tuning alone", delta_color="off")

tuning_table = pd.DataFrame({
    "depth": table["depth"] if "depth" in table.columns else table["sample"],
    "twt": table["twt"] if "twt" in table.columns else np.nan,
    "A_untuned": A_thick, "B_untuned": B_thick,
    "A_tuned": A_tuned, "B_tuned": B_tuned,
    "class_untuned": table["avo_class"].to_numpy(),
    "class_tuned": class_tuned,
    "dB_tuning": B_tuned - B_thick,
    "changes_class": changed,
}).round(4)
st.dataframe(tuning_table, use_container_width=True, hide_index=True)
st.caption(
    "`A_untuned` is the interface fit carried through the wavelet, so both "
    "columns are in the same units and a thick bed reads identically in each. "
    "Any difference is interference, not scaling."
)

fig = go.Figure()
for label, A_, B_, dash in (("untuned (interface)", A_thick, B_thick, None),
                            ("tuned (as picked)", A_tuned, B_tuned, "dot")):
    fig.add_trace(go.Scatter(
        x=A_, y=B_, mode="markers", name=label,
        marker=dict(size=12, symbol="circle" if dash is None else "diamond",
                    line=dict(width=1, color="#fff")),
    ))
for k in range(len(table)):
    if np.all(np.isfinite([A_thick[k], B_thick[k], A_tuned[k], B_tuned[k]])):
        fig.add_annotation(x=A_tuned[k], y=B_tuned[k], ax=A_thick[k], ay=B_thick[k],
                           xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=2, arrowwidth=1.3,
                           arrowcolor="#888", opacity=0.8)
fig.add_hline(y=0, line=dict(color="#666", width=1))
fig.add_vline(x=0, line=dict(color="#666", width=1))
fig.update_layout(xaxis_title="Intercept A", yaxis_title="Gradient B", height=560,
                  margin=dict(l=60, r=20, t=40, b=50),
                  legend=dict(orientation="h", yanchor="bottom", y=1.02))
st.plotly_chart(fig, use_container_width=True)
st.caption("Each arrow is what tuning does to that reflector on this well.")

# ------------------------------------------------- gather, class-coloured --
st.divider()
st.subheader("Gather, reflectors coloured by class")

gather = detail_gather
mode = st.radio("Display", ["Variable density", "Wiggle"], horizontal=True,
                key="class_gather_mode")
markers = table[["twt", "avo_class"]] if "twt" in table.columns else None
st.plotly_chart(gather_figure(gather, angles, twt, mode=mode, markers=markers),
                use_container_width=True)
st.caption("Triangles on the left edge mark classified reflectors.")


# ---------------------------------------------------- fluid-case compare ----
if well.has_fluid_cases:
    st.divider()
    st.header("Fluid case comparison")
    st.caption(
        "Every case is fitted at the **same** interfaces, on a time axis "
        f"integrated once from the *{well.active_case}* case. Without that, each "
        "case would carry its own time axis and the reflectors would not line up."
    )

    reference = st.selectbox(
        "Reference case", list(well.cases),
        index=well.cases.index("brine") if "brine" in well.cases else 0,
        help="The fluid vectors are drawn from this case to each of the others.",
    )

    case_tables = {}
    failed = []
    for case_name in well.cases:
        try:
            frame = time_well(well, settings, case_name)
        except ValueError:
            failed.append(case_name)
            continue
        case_rc = reflectivity_series(
            frame["VP"].to_numpy(float), frame["VS"].to_numpy(float),
            frame["RHOB"].to_numpy(float), angles, method=settings.method,
        )
        case_tables[case_name] = reflector_avo(
            case_rc, angles=angles, method=settings.method,
            depth=frame["DEPTH"].to_numpy(float) if "DEPTH" in frame.columns else None,
            twt=frame["TWT"].to_numpy(float),
            samples=table["sample"].to_numpy(), a_tol=settings.a_tol,
        )
    if failed:
        st.warning(f"Skipped case(s) with no usable samples: {', '.join(failed)}.")

    comparison = compare_cases(case_tables, reference=reference, a_tol=settings.a_tol)
    targets = [c for c in well.cases if c != reference and c in case_tables]

    changed = int(comparison["class_changed"].sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Reflectors compared", len(comparison))
    c2.metric("Change class with fluid", changed,
              f"{changed / max(len(comparison), 1):.0%} of reflectors")
    c3.metric("Largest fluid vector", f"{comparison['fluid_vector'].max():.4f}")

    st.plotly_chart(
        fluid_vector_crossplot(comparison, reference, targets, a_tol=settings.a_tol),
        use_container_width=True,
    )
    st.caption(
        f"Circles are the **{reference}** case; diamonds are the substituted cases, "
        "with an arrow along each reflector's fluid vector. A long arrow crossing a "
        "class boundary is a reflector whose AVO signature depends on what is in the "
        "pore space — the ones worth trusting a fluid interpretation on."
    )

    show_cmp = comparison.copy()
    for col in show_cmp.columns:
        if show_cmp[col].dtype.kind == "f":
            show_cmp[col] = show_cmp[col].round(4)
    st.dataframe(show_cmp, use_container_width=True, height=320)
    st.download_button(
        "Download fluid comparison (CSV)", show_cmp.to_csv(index=False).encode(),
        file_name=f"{well.name}_fluid_case_comparison.csv", mime="text/csv",
    )

    # Read the class path in fluid order — brine to oil to gas — with the
    # in-situ case noted at the end rather than interleaved.
    fluid_order = [c for c in ("brine", "oil", "gas") if c in case_tables]
    fluid_order += [c for c in case_tables if c not in fluid_order]

    movers = comparison[comparison["class_changed"]]
    if len(movers):
        st.subheader("Reflectors that change class")
        for _, row in movers.iterrows():
            where = (f"{row['depth']:.1f} m" if "depth" in comparison.columns
                     else f"sample {int(row['sample'])}")
            path = "  →  ".join(
                f"**{c}** {row[f'class_{c}']}" for c in fluid_order if c != "in situ"
            )
            if "in situ" in case_tables:
                path += f"   (in situ: {row['class_in situ']})"
            st.write(f"- {where} · {path}")
    else:
        st.info("No reflector changes AVO class across the fluid cases.")

    st.subheader("Amplitude vs angle, by fluid case")
    # This dropdown's labels are volatile too, but nothing reads its key before
    # the widget is created, so Streamlit's own validation repairs a value it
    # no longer recognises. The detail dropdown above is the one that needs
    # help, precisely because the page reads it first.
    pick_cmp = st.selectbox(
        "Reflector", range(len(comparison)),
        format_func=lambda i: (
            f"{comparison.iloc[i]['depth']:.1f} m"
            if "depth" in comparison.columns else f"sample {int(comparison.iloc[i]['sample'])}"
        ) + (" — class changes" if comparison.iloc[i]["class_changed"] else ""),
        key="fluid_reflector",
    )
    row_cmp = comparison.iloc[pick_cmp]
    fig = go.Figure()
    for case_name in fluid_order:
        a_c, b_c = row_cmp[f"A_{case_name}"], row_cmp[f"B_{case_name}"]
        fig.add_trace(go.Scatter(
            x=fine, y=a_c + b_c * sin2_fine, mode="lines",
            name=f"{case_name} — class {row_cmp[f'class_{case_name}']}",
            line=dict(width=2.4, color=case_colour(case_name)),
        ))
    fig.add_hline(y=0, line=dict(color="#bbb", width=1))
    fig.update_layout(xaxis_title="Incidence angle (deg)", yaxis_title="Rpp",
                      height=440, margin=dict(l=60, r=20, t=30, b=45),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig, use_container_width=True)
