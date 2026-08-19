"""Page 3 — intercept–gradient analysis and AVO classification of reflectors."""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from avo_qi.core.avo import background_trend, reflector_avo  # noqa: E402
from avo_qi.core.reflectivity import (  # noqa: E402
    aki_richards_rpp,
    reflectivity_series,
    zoeppritz_rpp,
)
from avo_qi.core.synthetic import build_gather  # noqa: E402
from avo_qi.ui import (  # noqa: E402
    CLASS_COLOURS,
    ab_crossplot,
    build_wavelet,
    gather_figure,
    page_setup,
    require_well,
    sidebar,
    time_well,
)

page_setup("AVO Classification", icon=":triangular_ruler:")
settings = sidebar()
well = require_well()

try:
    tw = time_well(well, settings)
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
table = reflector_avo(
    rc, vp, vs, rho, angles, method=settings.method, both=True,
    depth=depth, twt=twt, threshold=settings.threshold, a_tol=settings.a_tol,
)

if table.empty:
    st.warning(
        f"No interface exceeds the |R| threshold of {settings.threshold:.3f}. "
        "Lower it in the sidebar."
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

def _label(row):
    where = f"{row['depth']:.1f} m" if "depth" in table.columns else f"sample {int(row['sample'])}"
    when = f" / {row['twt']:.3f} s" if "twt" in table.columns else ""
    return f"{where}{when} — Class {row['avo_class']} (A {row['A_shuey']:+.3f}, B {row['B_shuey']:+.3f})"

labels = [_label(r) for _, r in table.iterrows()]
# Open on the strongest reflector rather than the shallowest, which on a noisy
# log is often a near-zero interface that happens to clear the threshold.
strongest = int(np.argmax(np.abs(table["R0"].to_numpy(float))))
pick = st.selectbox("Reflector", range(len(labels)), index=strongest,
                    format_func=lambda i: labels[i])
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

fig.add_hline(y=0, line=dict(color="#bbb", width=1))
fig.update_layout(xaxis_title="Incidence angle (deg)", yaxis_title="Rpp",
                  height=460, margin=dict(l=60, r=20, t=30, b=45),
                  legend=dict(orientation="h", yanchor="bottom", y=1.02))
st.plotly_chart(fig, use_container_width=True)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Class", row["avo_class"])
c2.metric("Intercept A", f"{row['A_shuey']:+.4f}")
c3.metric("Gradient B", f"{row['B_shuey']:+.4f}")
c4.metric("ΔA Shuey−AkiR", f"{row['dA']:+.1e}")
c5.metric("ΔB Shuey−AkiR", f"{row['dB']:+.1e}")

if i + 1 < vp.size:
    st.caption(
        f"Upper layer Vp {vp[i]:.0f} m/s, Vs {vs[i]:.0f} m/s, ρ {rho[i]:.3f} g/cc "
        f"(Vp/Vs {vp[i] / vs[i]:.2f}) over lower layer Vp {vp[i + 1]:.0f} m/s, "
        f"Vs {vs[i + 1]:.0f} m/s, ρ {rho[i + 1]:.3f} g/cc "
        f"(Vp/Vs {vp[i + 1] / vs[i + 1]:.2f})."
    )

# ------------------------------------------------- gather, class-coloured --
st.divider()
st.subheader("Gather, reflectors coloured by class")

_, wavelet = build_wavelet(settings)
gather = build_gather(vp, vs, rho, angles, wavelet, dt=settings.dt, method=settings.method)
mode = st.radio("Display", ["Variable density", "Wiggle"], horizontal=True,
                key="class_gather_mode")
markers = table[["twt", "avo_class"]] if "twt" in table.columns else None
st.plotly_chart(gather_figure(gather, angles, twt, mode=mode, markers=markers),
                use_container_width=True)
st.caption("Triangles on the left edge mark classified reflectors.")
