"""Page 6 — compare the wells in the library against each other.

Everything else in the toolkit works on one well. This is where more than one
becomes the point: a background trend is only a *background* if it is fitted
through more than one hole, a compaction trend needs several, and a class that
looks anomalous in one well may be the norm in the field.

Measured depth is useless here — it starts at a rig floor — so every
cross-well plot is against TVDSS or TVDBML. A well without a vertical
reference is named and left out rather than silently drawn at the wrong depth.
"""

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

from avo_qi.analysis import reflector_analysis  # noqa: E402
from avo_qi.core.avo import background_trend  # noqa: E402
from avo_qi.core.zones import zone_event_summary  # noqa: E402
from avo_qi.ui import (  # noqa: E402
    CLASS_COLOURS,
    DEPTH_REFERENCES,
    active_well_name,
    avo_attribute_panel,
    class_depth_panel,
    class_property_panel,
    page_setup,
    sidebar,
    well_settings_for,
    wells,
)
from avo_qi.ui_colours import well_colour  # noqa: E402

page_setup("Multi-well", icon=":material/compare_arrows:")
settings = sidebar(show_classifier=True)

library = wells()
if len(library) < 2:
    st.warning(
        f"{len(library)} well loaded. Load a second one on the **Load & QC** "
        "page — this page compares wells, and one well is the rest of the "
        "toolkit.", icon=":material/info:")
    st.stop()

# ------------------------------------------------------------- selection ---
st.subheader("Wells to compare")
chosen = st.multiselect("Wells", list(library), default=list(library),
                        key="multiwell_pick")
if len(chosen) < 2:
    st.info("Pick at least two.")
    st.stop()

st.caption(
    "Each well is analysed on **its own** active fluid case and its own "
    "tops and datum, with the shared physics from the sidebar — the same "
    "sample rate, angles, wavelet, amplitude cut and class tolerance for all "
    "of them. Comparing wells fitted with different physics would compare the "
    "settings, not the rock."
)

# The analysis is expensive on a real well, so it runs on request and is kept
# with the settings it was run under; changing those settings makes the stored
# result stale rather than silently wrong.
def _signature():
    return (settings.dt, settings.t0, settings.method, settings.threshold,
            settings.a_tol, settings.wavelet_kind, settings.ricker_freq,
            tuple(settings.ormsby), settings.wavelet_length,
            float(settings.angle_min), float(settings.angle_max),
            float(settings.angle_step), tuple(chosen))


run = st.button("Compare wells", type="primary")
stored = st.session_state.get("multiwell_results")
if run:
    results, failed = {}, {}
    progress = st.progress(0.0, "Analysing…")
    for i, name in enumerate(chosen, start=1):
        progress.progress(i / len(chosen), f"Analysing {name}…")
        try:
            found = reflector_analysis(library[name], well_settings_for(name))
        except ValueError as exc:                 # a well that cannot be run
            failed[name] = str(exc)               # must not stop the others
            continue
        results[name] = found
    progress.empty()
    stored = {"results": results, "failed": failed, "signature": _signature()}
    st.session_state["multiwell_results"] = stored

if not stored:
    st.info("Press **Compare wells** to run the reflector analysis on each.",
            icon=":material/play_arrow:")
    st.stop()

if stored["signature"] != _signature():
    st.warning(
        "The settings have changed since this was run, so what is below was "
        "computed with the previous ones. Press **Compare wells** again.",
        icon=":material/warning:")

results = {n: r for n, r in stored["results"].items() if n in chosen}
for name, why in stored.get("failed", {}).items():
    st.error(f"{name}: {why}")
if not results:
    st.stop()

colours = {name: well_colour(i) for i, name in enumerate(results)}

# ------------------------------------------------------------- overview ----
rows = []
for name, found in results.items():
    table = found["table"]
    frame = library[name].df
    references = [c for c in DEPTH_REFERENCES if c in frame.columns]
    rows.append({
        "well": name,
        "case": found["case"],
        "events": len(table),
        "class III": int((table["avo_class"] == "III").sum()) if len(table) else 0,
        "median |A|": (round(float(np.nanmedian(np.abs(table["A_shuey"]))), 4)
                       if len(table) else np.nan),
        "vertical reference": ", ".join(references) or "—",
        "interval (m MD)": (f"{frame['DEPTH'].min():,.0f}–"
                            f"{frame['DEPTH'].max():,.0f}"),
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# --------------------------------------------------- intercept–gradient ----
st.divider()
st.subheader("Intercept–gradient, all wells")

# Each well's events carry their distance from **that well's own** trend, not
# from the shared one. A trend is what the ordinary rock in a hole does, and
# measuring well B's anomalies against well A's background would report the
# difference between the two wells as an anomaly in every reflector of one.
def _with_own_deviation(table, name):
    own = background_trend(table["A_shuey"], table["B_shuey"])
    deviation = (own.deviation(table["A_shuey"], table["B_shuey"])
                 if np.isfinite(own.slope) else np.nan)
    return table.assign(well=name, background_deviation=deviation)


everything = pd.concat(
    [_with_own_deviation(found["table"], name)
     for name, found in results.items()], ignore_index=True)
shared = background_trend(everything["A_shuey"], everything["B_shuey"])

fig = go.Figure()
a_all = everything["A_shuey"].to_numpy(float)
b_all = everything["B_shuey"].to_numpy(float)
limit = max(float(np.nanmax(np.abs(a_all))) * 1.2, 0.05) if a_all.size else 0.25
fig.add_hline(y=0, line=dict(color="#666", width=1))
fig.add_vline(x=0, line=dict(color="#666", width=1))
if np.isfinite(shared.slope):
    line = np.array([-limit, limit])
    fig.add_trace(go.Scatter(
        x=line, y=shared.slope * line + shared.intercept, mode="lines",
        name=f"trend, all wells (n={shared.n_points})",
        line=dict(color="#20242A", width=2.5, dash="dash")))
for name, found in results.items():
    table = found["table"]
    if not len(table):
        continue
    where = ("depth" if "depth" in table.columns else "sample")
    fig.add_trace(go.Scatter(
        x=table["A_shuey"], y=table["B_shuey"], mode="markers", name=name,
        marker=dict(size=9, color=colours[name], line=dict(width=0.8,
                                                           color="#FFFFFF")),
        text=[f"{name} · {v:,.1f} m · {c}" for v, c in
              zip(table[where], table["avo_class"])],
        hovertemplate="A %{x:.4f}<br>B %{y:.4f}<br>%{text}<extra></extra>"))
fig.update_layout(xaxis_title="Intercept A", yaxis_title="Gradient B",
                  height=620, margin=dict(l=60, r=20, t=30, b=50),
                  legend=dict(orientation="h", yanchor="bottom", y=1.02))
st.plotly_chart(fig, use_container_width=True)

trends = []
for name, found in results.items():
    own = background_trend(found["table"]["A_shuey"], found["table"]["B_shuey"])
    trends.append({"well": name, "slope": round(own.slope, 3),
                   "intercept": round(own.intercept, 4), "events": own.n_points})
trends.append({"well": "— all wells —", "slope": round(shared.slope, 3),
               "intercept": round(shared.intercept, 4),
               "events": shared.n_points})
st.dataframe(pd.DataFrame(trends), use_container_width=True, hide_index=True)
st.caption(
    "A background trend fitted in one well is that well's rock; fitted across "
    "several it is the field's, and an anomaly measured against it means "
    "something wider. Where one well's own slope departs from the shared one, "
    "that is worth understanding before trusting either — different "
    "compaction, a different lithology mix, or a well that simply has few "
    "events to fit."
)

# ------------------------------------------------------ depth behaviour ----
st.divider()
st.subheader("Against true vertical depth")

available = [r for r in ("TVDSS", "TVDBML")
             if any(r in library[n].df.columns for n in results)]
if not available:
    st.info(
        "No well here has a vertical reference, so there is nothing to plot "
        "against: measured depth starts at a rig floor and is not comparable "
        "between wells. Set the datum on the **Load & QC** page.",
        icon=":material/info:")
else:
    c1, c2 = st.columns(2)
    reference = c1.selectbox("Depth reference", available,
                             help="TVDSS compares wells at the same depth in "
                                  "the earth; TVDBML follows compaction from "
                                  "the sea bed, which is what porosity and "
                                  "velocity actually track.")
    curves = [c for c in ("PHI", "VSH", "SW", "VP", "VS", "RHOB")
              if any(c in library[n].df.columns for n in results)]
    curve = c2.selectbox("Curve", curves)

    missing = [n for n in results if reference not in library[n].df.columns]
    fig = go.Figure()
    for name in results:
        frame = library[name].df
        if reference not in frame.columns or curve not in frame.columns:
            continue
        y = frame[reference].to_numpy(float)
        x = frame[curve].to_numpy(float)
        good = np.isfinite(x) & np.isfinite(y)
        # Real logs are tens of thousands of samples; drawing every one of them
        # per well makes the browser, not the geology, the bottleneck.
        step = max(int(good.sum() // 4000), 1)
        fig.add_trace(go.Scattergl(
            x=x[good][::step], y=y[good][::step], mode="markers", name=name,
            marker=dict(size=3, color=colours[name], opacity=0.45)))
    fig.update_yaxes(autorange="reversed", title_text=f"{reference} (m)")
    fig.update_layout(xaxis_title=curve, height=640,
                      margin=dict(l=70, r=20, t=30, b=50),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig, use_container_width=True)
    if missing:
        st.caption(", ".join(f"**{n}**" for n in missing)
                   + f" has no {reference} and is left out — drawing it at its "
                     "measured depth would put it at the wrong depth in the "
                     "earth, which is the error this reference exists to stop.")


# ------------------------------------------------- where the classes are ----
# Replaces a single A-against-TVDSS plot that used to live in the depth
# section above: the same events, the same colouring, but with the class on an
# axis of its own and a panel per property, all sharing one depth axis.
st.divider()
st.subheader("Where the classes are, all wells")
st.caption(
    "Every panel shares the depth axis, so a horizontal line across the figure "
    "is one reflector; colour is the class and shape is the well. A class that "
    "clusters at one depth **across** wells is a rock property; one that "
    "appears at a different depth in each is more likely a fluid or a facies "
    "change. Wells with no vertical reference cannot be drawn here at all — "
    "measured depth would put them at the wrong depth in the earth."
)
class_depth_panel(everything, key="multiwell_class_depth", split_column="well")

# --------------------------------------------------------- class by well ---
st.divider()
st.subheader("What each well is made of")

mix = []
for name, found in results.items():
    table = found["table"]
    counts = table["avo_class"].value_counts() if len(table) else {}
    row = {"well": name, "events": len(table)}
    for label in CLASS_COLOURS:
        row[label] = int(counts.get(label, 0))
    mix.append(row)
st.dataframe(pd.DataFrame(mix), use_container_width=True, hide_index=True)

zoned = []
for name, found in results.items():
    table = found["table"]
    if "zone" not in table.columns:
        continue
    summary = zone_event_summary(table, zone_columns=("zone", "zone_below"))
    zoned.append(summary.assign(well=name))
if zoned:
    st.caption("Events by zone, for the wells that carry a zonation.")
    by_zone = pd.concat(zoned, ignore_index=True)
    st.dataframe(by_zone[["well"] + [c for c in by_zone.columns if c != "well"]],
                 use_container_width=True, hide_index=True)

# -------------------------------------------------------- AVO attributes ----
st.divider()
st.subheader("AVO attributes, all wells")
st.caption(
    "Every well's reflectors ranked together, each scored against **its own** "
    "background — the trend fitted through that hole and the scatter of that "
    "hole's own deviations. Pooling the scaling instead would let the noisiest "
    "well set the yardstick for all of them and bury a quiet well's best "
    "event. The rank still spans the wells, because *what should I look at "
    "first* is one question across a field."
)
avo_attribute_panel(everything, shared, key="multiwell_attributes",
                    split_column="well")

# ------------------------------------------------- class against property ---
# The same panel page 4 carries, over every well at once. This is where it
# earns its keep: a class-porosity separation seen in one well is that well's
# rock and might be a coincidence of five reflectors; seen across the field it
# is a relationship worth predicting away from a well with.
st.divider()
st.subheader("Class against property, all wells")
st.caption(
    "Every compared well's reflectors pooled, each event's properties averaged "
    "over the **same two half-lobes its own intercept and gradient were fitted "
    "from**. Marker shape is the well, so a class carried by one hole alone "
    "reads as that rather than as a property of the field. The wells must "
    "share a petrophysical definition for this to mean anything — the net "
    "cutoffs in the sidebar are one definition for all of them, but VSH and "
    "PHI computed by different vendors on different tools are not."
)
class_property_panel(everything, key="multiwell_property", split_column="well")

st.divider()
st.download_button(
    "Download every well's reflectors (CSV)",
    everything.drop(columns=["props_row"], errors="ignore").to_csv(index=False).encode(),
    file_name="multi_well_reflectors.csv", mime="text/csv")
st.caption(
    f"Active well is **{active_well_name()}**; switching it in the sidebar "
    "changes the other pages, not this one."
)
