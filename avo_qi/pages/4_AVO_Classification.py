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

from avo_qi.core.blocking import blocked_reflectivity, half_cycle_samples  # noqa: E402
from avo_qi.core.lithology import interface_lithology  # noqa: E402
from avo_qi.core.tuning import (  # noqa: E402
    apparent_period,
    tuned_amplitudes,
    tuning_thickness_from_wavelet,
)
from avo_qi.core.avo import (  # noqa: E402
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
from avo_qi.ui import (  # noqa: E402
    CLASS_COLOURS,
    ab_crossplot,
    lithology_labels,
    build_wavelet,
    case_colour,
    classified_trace_figure,
    fluid_vector_crossplot,
    gather_figure,
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

rc = reflectivity_series(vp, vs, rho, angles, method=settings.method)

# --- how the untuned response is measured ------------------------------------
st.subheader("Untuned response")
c1, c2, c3 = st.columns([2, 1, 1])
blocking_mode = c1.radio(
    "Layer properties from",
    ["Half-cycle blocked layers", "Adjacent samples"],
    horizontal=True,
    help="Two adjacent samples carry the true layer contrast only when the "
         "boundary is a step. On a gradational boundary the contrast splits "
         "across several samples and no single interface carries it.",
)
use_blocking = blocking_mode.startswith("Half-cycle")
block_method = c2.selectbox("Average", ["backus", "mean"], disabled=not use_blocking,
                            help="Backus is the correct elastic upscaling; the "
                                 "arithmetic mean is easier to read but is not "
                                 "what a wave does.")
guard = c3.number_input("Guard (samples)", 0, 20, 2, 1, disabled=not use_blocking,
                        help="Skip this many samples either side of the boundary "
                             "so a gradational ramp is excluded from both averages.")

_, page_wavelet = build_wavelet(settings)
window = half_cycle_samples(apparent_period(page_wavelet, settings.dt), settings.dt)
tuning_twt = tuning_thickness_from_wavelet(page_wavelet, settings.dt)

table = reflector_avo(
    rc, vp, vs, rho, angles, method=settings.method, both=True,
    depth=depth, twt=twt, threshold=settings.threshold, a_tol=settings.a_tol,
)

if use_blocking and not table.empty:
    adjacent = table[["A_shuey", "B_shuey", "avo_class"]].rename(
        columns={"A_shuey": "A_adjacent", "B_shuey": "B_adjacent",
                 "avo_class": "class_adjacent"}).reset_index(drop=True)
    blocked = blocked_reflectivity(
        vp, vs, rho, table["sample"].to_numpy(), angles, window=window,
        method=block_method, guard=int(guard), reflectivity_method=settings.method,
    )
    rc_blocked = np.zeros_like(rc)
    rc_blocked[table["sample"].to_numpy(), :] = np.nan_to_num(blocked["rc"], nan=0.0)
    blocked_mask = np.isnan(blocked["rc"])

    rc_for_fit = rc_blocked.copy()
    rc_for_fit[table["sample"].to_numpy(), :] = blocked["rc"]
    table = reflector_avo(
        rc_for_fit, vp, vs, rho, angles, method=settings.method, both=True,
        depth=depth, twt=twt, samples=table["sample"].to_numpy(),
        a_tol=settings.a_tol, mask_post_critical=False,
    )
    table["critical_angle"] = blocked["critical_angle"]
    table["n_angles"] = (~blocked_mask).sum(axis=1)
    table = pd.concat([table, adjacent], axis=1)
    table["dA_blocking"] = table["A_shuey"] - table["A_adjacent"]

    moved = int((table["avo_class"] != table["class_adjacent"]).sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Blocking window", f"±{window} samples",
              f"{window * settings.dt * 1000:.0f} ms — a half cycle")
    c2.metric("Largest change in A", f"{table['dA_blocking'].abs().max():+.4f}")
    c3.metric("Reflectors that change class", moved,
              delta=None if moved == 0 else "vs adjacent samples",
              delta_color="off")
    if moved:
        st.info(
            f"{moved} reflector(s) classify differently once each side is averaged "
            "over a half cycle. On a gradational boundary the adjacent-sample "
            "coefficient is the weaker, not the truer, of the two.",
            icon=":material/info:",
        )
else:
    st.caption(
        "Using two adjacent log samples per interface. Exact for a blocky log; "
        "on a gradational boundary the contrast splits across samples and this "
        "under-reads it."
    )
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

selected_litho = set(settings.lithologies or [])
if selected_litho:
    keep = np.array([(u in selected_litho) or (lo in selected_litho)
                     for u, lo in zip(pairs["upper"], pairs["lower"])], dtype=bool)
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
            "*sand over* ones. Filtering to shale-over-sand is usually the "
            "quickest way to a clean A-B cloud."
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

# --------------------------------------------------------------- table -----
st.divider()
st.subheader("Reflector table")

show = table.copy()
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
filtered = show[show["avo_class"].isin(class_filter)]
st.dataframe(filtered, use_container_width=True, height=340)

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
    "Click a marker on the trace to inspect that reflector. Each marker sits on "
    "the amplitude extremum the reflector produces, coloured by its AVO class."
)

# The gather this page's trace is drawn from.
_, detail_wavelet = build_wavelet(settings)
detail_gather = build_gather(vp, vs, rho, angles, detail_wavelet,
                             dt=settings.dt, method=settings.method)

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
if "reflector_pick" not in st.session_state:
    st.session_state["reflector_pick"] = strongest

# A click on the trace arrives in this run's session state, before the
# selectbox below is drawn, so reading it here lets the click drive the
# dropdown instead of fighting it.  Only act on a *new* selection, or manual
# changes to the dropdown would be overridden on every rerun.
clicked = selected_reflector_index(st.session_state.get("class_trace"), marker_twt)
if clicked is not None and clicked != st.session_state.get("_last_trace_click"):
    st.session_state["_last_trace_click"] = clicked
    st.session_state["reflector_pick"] = clicked
if st.session_state["reflector_pick"] >= len(labels):
    st.session_state["reflector_pick"] = strongest

trace_col, detail_col = st.columns([1, 2])

with trace_col:
    st.markdown(f"**{trace_choice}** — reflectors by class")
    st.plotly_chart(
        classified_trace_figure(
            detail_trace, twt, table, extrema,
            selected=st.session_state["reflector_pick"],
        ),
        use_container_width=True, key="class_trace",
        on_select="rerun", selection_mode="points",
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

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=angles, y=rc[i, :], mode="markers", name=f"{settings.method.replace('_', '-')} (modelled)",
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

    # The two exact interface models, for reference.
    if i + 1 < vp.size:
        layer = (vp[i], vs[i], rho[i], vp[i + 1], vs[i + 1], rho[i + 1])
        fig.add_trace(go.Scatter(x=fine, y=zoeppritz_rpp(*layer, fine), mode="lines",
                                 name="Zoeppritz", line=dict(width=1, color="#999")))
        fig.add_trace(go.Scatter(x=fine, y=aki_richards_rpp(*layer, fine), mode="lines",
                                 name="Aki-Richards (3-term)",
                                 line=dict(width=1, dash="dot", color="#999")))

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
    fig.update_layout(xaxis_title="Incidence angle (deg)", yaxis_title="Rpp",
                      height=460, margin=dict(l=60, r=20, t=30, b=45),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig, use_container_width=True)

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

    if i + 1 < vp.size:
        pair = row.get("litho_pair", "")
        if pair and "undefined" not in str(pair):
            st.caption(f"Lithology: **{pair}**.")
        st.caption(
            f"Upper layer Vp {vp[i]:.0f} m/s, Vs {vs[i]:.0f} m/s, ρ {rho[i]:.3f} g/cc "
            f"(Vp/Vs {vp[i] / vs[i]:.2f}) over lower layer Vp {vp[i + 1]:.0f} m/s, "
            f"Vs {vs[i + 1]:.0f} m/s, ρ {rho[i + 1]:.3f} g/cc "
            f"(Vp/Vs {vp[i + 1] / vs[i + 1]:.2f})."
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
