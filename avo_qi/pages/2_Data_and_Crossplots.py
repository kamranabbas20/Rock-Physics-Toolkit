"""Page 1 — load a well, remap mnemonics, and draw the QI crossplots."""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402

from avo_qi.core.attributes import eei  # noqa: E402
from avo_qi.ui import (  # noqa: E402
    add_derived_curves,
    apply_lithology_filter,
    case_colour,
    crossplot,
    lithology_crossplot,
    lithology_labels,
    log_track_figure,
    page_setup,
    require_well,
    sidebar,
)

page_setup("Data & Crossplots", icon=":bar_chart:")
settings = sidebar(show_wavelet=False, show_angles=False, show_classifier=False)
well = require_well()

if st.session_state.get("raw_df") is not None:
    st.caption(
        "Loading, curve assignment and QC live on the **Load & QC** page — "
        "this page plots whatever that one hands over."
    )

# ----------------------------------------------------------- attributes ----
chi = st.session_state.get("chi_deg", 0.0)
df = add_derived_curves(well.complete(settings.case).reset_index(drop=True))

if well.has_fluid_cases:
    st.info(
        f"This well carries **{len(well.cases)} fluid cases** — "
        f"{', '.join(well.cases)}. Showing **{settings.case}**; switch in the "
        "sidebar, or overlay them all on the crossplots below.",
        icon=":material/water_drop:",
    )

st.divider()
st.subheader("Log tracks")
available = [c for c in ("VP", "VS", "RHOB", "AI", "SI", "VPVS", "POISSON", "GR", "VSH",
                         "PHI", "SW", "LAMBDA_RHO", "MU_RHO") if c in df.columns]
tracks = st.multiselect("Curves to display", available,
                        default=[c for c in ("VP", "VS", "RHOB", "AI", "VPVS")
                                 if c in available])
if tracks:
    st.plotly_chart(log_track_figure(df, curves=tracks), use_container_width=True)

with st.expander("Curve statistics"):
    st.dataframe(df[available].describe().T, use_container_width=True)

# ----------------------------------------------------------- crossplots ----
st.divider()
st.subheader("QI crossplots")

litho = lithology_labels(df, settings)
keep = apply_lithology_filter(df, litho, settings)
if not keep.all():
    hidden = int((~keep).sum())
    st.caption(f"Lithology filter is hiding {hidden} of {len(df)} samples "
               f"({', '.join(settings.lithologies) or 'nothing selected'}).")
df = df[keep].reset_index(drop=True)
litho = litho[keep]
if df.empty:
    st.warning("The lithology filter has excluded every sample. Widen it in the sidebar.")
    st.stop()

colour_options = ["LITHOLOGY"] + [c for c in ("DEPTH", "GR", "VSH", "PHI", "SW",
                                              "VPVS", "POISSON", "FACIES")
                                  if c in df.columns]
c1, c2 = st.columns([2, 1])
colour = c1.selectbox("Colour by", colour_options, index=0)
overlay_cases = c2.toggle(
    "Overlay all fluid cases", value=False, disabled=not well.has_fluid_cases,
    help="Plot every substituted case together, coloured by case instead of by curve.",
) if well.has_fluid_cases else False


def qi_plot(x, y, title, log_x=False):
    """One crossplot: by lithology, by a curve, or split by fluid case."""
    if not overlay_cases:
        if colour == "LITHOLOGY":
            return lithology_crossplot(df, litho, x, y, title=title)
        return crossplot(df, x, y, colour, title=title, log_x=log_x)
    import plotly.graph_objects as go

    fig = go.Figure()
    for case in well.cases:
        sub = add_derived_curves(well.complete(case).reset_index(drop=True))
        fig.add_trace(go.Scatter(
            x=sub[x], y=sub[y], mode="markers", name=case,
            marker=dict(size=4, opacity=0.7, color=case_colour(case)),
            hovertemplate=f"{x}: %{{x:.4g}}<br>{y}: %{{y:.4g}}<extra>{case}</extra>",
        ))
    fig.update_layout(title=title, xaxis_title=x, yaxis_title=y, height=520,
                      margin=dict(l=60, r=130, t=50, b=50),
                      legend=dict(orientation="v", yanchor="top", y=1.0, x=1.02))
    if log_x:
        fig.update_xaxes(type="log")
    return fig

tab_ai, tab_lmr, tab_ipis, tab_pr, tab_eei = st.tabs(
    ["AI vs Vp/Vs", "λρ – μρ (LMR)", "IP – IS", "Poisson vs AI", "EEI"]
)

with tab_ai:
    st.plotly_chart(
        qi_plot("AI", "VPVS", title="Vp/Vs vs acoustic impedance"),
        use_container_width=True,
    )
    st.caption(
        "Gas sands separate towards low AI and low Vp/Vs — the lower-left of this plot."
    )

with tab_lmr:
    st.plotly_chart(
        qi_plot("LAMBDA_RHO", "MU_RHO", title="LMR: μρ vs λρ"),
        use_container_width=True,
    )
    st.caption("λρ is the fluid-sensitive axis; μρ responds to the rock frame.")

with tab_ipis:
    st.plotly_chart(
        qi_plot("AI", "SI", title="S-impedance vs P-impedance"),
        use_container_width=True,
    )

with tab_pr:
    st.plotly_chart(
        qi_plot("AI", "POISSON", title="Poisson's ratio vs acoustic impedance"),
        use_container_width=True,
    )

with tab_eei:
    chi = st.slider("χ angle (deg)", -90.0, 90.0, float(chi), 1.0,
                    help="χ = 0 reproduces acoustic impedance; χ near 30–40° tends to "
                         "track fluid, χ near -50° tends to track lithology.")
    st.session_state["chi_deg"] = chi
    df["EEI"] = eei(df["VP"].to_numpy(float), df["VS"].to_numpy(float),
                    df["RHOB"].to_numpy(float), chi)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(crossplot(df, "EEI", "VPVS", colour,
                                  title=f"Vp/Vs vs EEI (χ = {chi:.0f}°)"),
                        use_container_width=True)
    with c2:
        st.plotly_chart(log_track_figure(df, curves=["EEI", "AI"], height=520),
                        use_container_width=True)

    targets = [c for c in ("SW", "VSH", "PHI") if c in df.columns]
    if targets:
        target = st.selectbox("Correlate EEI against", targets)
        chis = np.arange(-90, 91, 2.0)
        values = df[target].to_numpy(float)
        corr = []
        for c in chis:
            e = eei(df["VP"].to_numpy(float), df["VS"].to_numpy(float),
                    df["RHOB"].to_numpy(float), c)
            good = np.isfinite(e) & np.isfinite(values)
            corr.append(np.corrcoef(e[good], values[good])[0, 1]
                        if good.sum() > 2 and np.std(values[good]) > 0 else np.nan)
        corr = np.array(corr)
        if np.any(np.isfinite(corr)):
            best = chis[int(np.nanargmax(np.abs(corr)))]
            st.caption(f"|correlation| with {target} peaks at χ = {best:.0f}°.")
            import plotly.graph_objects as go

            fig = go.Figure(go.Scatter(x=chis, y=corr, mode="lines"))
            fig.add_vline(x=best, line=dict(color="#d62728", dash="dash"))
            fig.update_layout(xaxis_title="χ (deg)", yaxis_title=f"corr(EEI, {target})",
                              height=300, margin=dict(l=60, r=20, t=20, b=40))
            st.plotly_chart(fig, use_container_width=True)

st.divider()
with st.expander("Standardised well table"):
    st.dataframe(df, use_container_width=True, height=400)
st.download_button("Download standardised well (CSV)", df.to_csv(index=False).encode(),
                   file_name=f"{well.name}_standardised.csv", mime="text/csv")
