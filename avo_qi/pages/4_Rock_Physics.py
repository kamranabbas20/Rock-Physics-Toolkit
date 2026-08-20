"""Page 4 — diagnostic rock-physics plots: bounds, trends and frame models.

Every curve here is a *model overlay* to diagnose the well against.  No
fluid substitution happens on this page or anywhere in the toolkit: the
mineral-plus-fluid bounds bracket the saturated rock directly, and the
granular frame models are labelled as dry-frame where they are shown.
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from avo_qi.core.rockphysics import (  # noqa: E402
    FLUIDS,
    MINERALS,
    bulk_modulus,
    castagna_mudrock,
    coordination_number,
    critical_porosity_dry,
    fit_gardner,
    gardner_density,
    greenberg_castagna,
    hashin_shtrikman,
    hertz_mindlin,
    raymer_hunt_gardner,
    shear_modulus,
    soft_sand_dry,
    stiff_sand_dry,
    velocities_from_moduli,
    voigt,
    reuss,
    wyllie,
)
from avo_qi.ui import add_derived_curves, page_setup, require_well, sidebar  # noqa: E402

page_setup("Rock Physics", icon=":rock:")
settings = sidebar(show_wavelet=False, show_angles=False, show_classifier=False)
well = require_well()

df = add_derived_curves(well.complete(settings.case).reset_index(drop=True))
vp = df["VP"].to_numpy(float)
vs = df["VS"].to_numpy(float)
rho = df["RHOB"].to_numpy(float)
has_phi = "PHI" in df.columns and np.isfinite(df["PHI"]).any()

if well.has_fluid_cases:
    st.info(
        f"Showing the **{settings.case}** case. The pore fluid below follows the "
        "case name where it can, so the bounds are drawn for the fluid the logs "
        "were substituted to.",
        icon=":material/water_drop:",
    )

st.caption(
    "Model overlays for diagnosing the well — mixture bounds, empirical trends "
    "and granular frame models. Nothing here substitutes fluids: the "
    "mineral-plus-fluid bounds bracket the saturated rock directly, and the "
    "granular models are dry-frame and marked as such."
)

# ------------------------------------------------------------ model setup ---
with st.expander("Model parameters", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    mineral_names = list(MINERALS)
    m1 = c1.selectbox("Mineral 1", mineral_names, index=mineral_names.index("quartz"))
    m2 = c1.selectbox("Mineral 2", mineral_names, index=mineral_names.index("clay"))
    frac1 = c2.slider(f"Fraction {m1}", 0.0, 1.0, 1.0, 0.05,
                      help=f"The remainder is {m2}.")
    # A substituted case names its own pore fluid, so follow it by default.
    fluid_options = list(FLUIDS)
    case_fluid = settings.case if settings.case in fluid_options else "brine"
    fluid_name = c2.selectbox(
        "Pore fluid", fluid_options, index=fluid_options.index(case_fluid),
        help="Defaults to the fluid named by the selected case in the sidebar.",
    )
    phi_c = c3.slider("Critical porosity φc", 0.20, 0.50, 0.36, 0.01)
    pressure_mpa = c3.slider("Effective pressure (MPa)", 1.0, 60.0, 10.0, 1.0)
    n_default = float(coordination_number(phi_c))
    n_grains = c4.slider("Coordination number n", 4.0, 16.0, round(n_default, 1), 0.1,
                         help=f"Murphy's relation gives {n_default:.1f} at φc = {phi_c:.2f}.")
    shear_factor = c4.slider("Shear factor f", 0.0, 1.0, 1.0, 0.05,
                             help="1 = perfect adhesion between grains, 0 = frictionless.")

K1, G1, rho1 = MINERALS[m1]
K2, G2, rho2 = MINERALS[m2]
K_min = voigt([K1, K2], [frac1, 1 - frac1])
G_min = voigt([G1, G2], [frac1, 1 - frac1])
K_min = 0.5 * (K_min + reuss([K1, K2], [frac1, 1 - frac1]))     # Hill average
G_min = 0.5 * (G_min + reuss([G1, G2], [frac1, 1 - frac1]))
rho_min = frac1 * rho1 + (1 - frac1) * rho2
K_fl, rho_fl = FLUIDS[fluid_name]
pressure = pressure_mpa * 1e6

c1, c2, c3, c4 = st.columns(4)
c1.metric("Mineral K", f"{K_min:.1f} GPa")
c2.metric("Mineral μ", f"{G_min:.1f} GPa")
c3.metric("Mineral ρ", f"{rho_min:.2f} g/cc")
K_hm, G_hm = hertz_mindlin(K_min, G_min, phi_c, n_grains, pressure, shear_factor)
c4.metric("Hertz-Mindlin K (dry)", f"{K_hm:.2f} GPa")

phi_grid = np.linspace(0.0, phi_c, 120)
colour_options = [c for c in ("DEPTH", "GR", "VSH", "PHI", "SW", "VPVS") if c in df.columns]
colour = st.selectbox("Colour the well data by", colour_options,
                      index=colour_options.index("GR") if "GR" in colour_options else 0)


def data_trace(x, y, name="Well data"):
    return go.Scatter(
        x=x, y=y, mode="markers", name=name,
        marker=dict(size=5, opacity=0.75, color=df[colour], colorscale="Viridis",
                    showscale=True, colorbar=dict(title=colour, x=1.02)),
        hovertemplate="%{x:.4g}, %{y:.4g}<extra></extra>",
    )


def finish(fig, xtitle, ytitle, height=560):
    fig.update_layout(
        xaxis_title=xtitle, yaxis_title=ytitle, height=height,
        margin=dict(l=60, r=20, t=40, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


tab_vpvs, tab_gardner, tab_vphi, tab_moduli = st.tabs(
    ["Vp – Vs trends", "Gardner", "Velocity – porosity", "Moduli & bounds"]
)

# ------------------------------------------------------------ Vp vs Vs ------
with tab_vpvs:
    fig = go.Figure()
    fig.add_trace(data_trace(vp, vs))
    grid = np.linspace(np.nanmin(vp) * 0.9, np.nanmax(vp) * 1.1, 100)
    fig.add_trace(go.Scatter(x=grid, y=castagna_mudrock(grid), mode="lines",
                             name="Castagna mudrock line",
                             line=dict(color="#d62728", width=2.5)))
    for lith, dash in (("sandstone", "dash"), ("shale", "dot"), ("limestone", "dashdot")):
        fig.add_trace(go.Scatter(x=grid, y=greenberg_castagna(grid, {lith: 1.0}),
                                 mode="lines", name=f"Greenberg-Castagna {lith}",
                                 line=dict(width=1.6, dash=dash)))
    st.plotly_chart(finish(fig, "Vp (m/s)", "Vs (m/s)"), use_container_width=True)
    st.caption(
        "A gas sand carries anomalously high Vs for its Vp, so it plots **above** "
        "the mudrock line; brine sands and shales sit on or near their trends."
    )

    fig2 = go.Figure()
    fig2.add_trace(data_trace(vp, df["VPVS"]))
    fig2.add_trace(go.Scatter(x=grid, y=grid / castagna_mudrock(grid), mode="lines",
                              name="Mudrock line", line=dict(color="#d62728", width=2.5)))
    st.plotly_chart(finish(fig2, "Vp (m/s)", "Vp/Vs", height=460), use_container_width=True)

# ------------------------------------------------------------- Gardner ------
with tab_gardner:
    a_fit, b_fit = fit_gardner(vp, rho)
    fig = go.Figure()
    fig.add_trace(data_trace(vp, rho))
    grid = np.linspace(np.nanmin(vp) * 0.9, np.nanmax(vp) * 1.1, 100)
    fig.add_trace(go.Scatter(x=grid, y=gardner_density(grid), mode="lines",
                             name="Gardner (0.31, 0.25)",
                             line=dict(color="#d62728", width=2.5)))
    if np.isfinite(a_fit):
        fig.add_trace(go.Scatter(x=grid, y=gardner_density(grid, a_fit, b_fit),
                                 mode="lines",
                                 name=f"Fitted ({a_fit:.3f}, {b_fit:.3f})",
                                 line=dict(color="#2ca02c", width=2, dash="dash")))
    st.plotly_chart(finish(fig, "Vp (m/s)", "Density (g/cc)"), use_container_width=True)
    if np.isfinite(a_fit):
        c1, c2 = st.columns(2)
        c1.metric("Fitted coefficient a", f"{a_fit:.3f}", f"{a_fit - 0.31:+.3f} vs Gardner")
        c2.metric("Fitted exponent b", f"{b_fit:.3f}", f"{b_fit - 0.25:+.3f} vs Gardner")
    st.caption(
        "Gardner's ρ = 0.31 Vp^0.25 is a clastic average. A well fitting a "
        "markedly lower exponent is often gas-affected or carbonate-rich."
    )

# -------------------------------------------------- velocity vs porosity ----
with tab_vphi:
    if not has_phi:
        st.info(
            "This well has no porosity curve, so only the model trends are drawn. "
            "Map a PHI curve on the **Data & Crossplots** page to overlay the data."
        )
    v_matrix, _ = velocities_from_moduli(K_min, G_min, rho_min)
    v_matrix = float(v_matrix)
    v_fluid = float(np.sqrt(K_fl * 1e6 / rho_fl))

    fig = go.Figure()
    if has_phi:
        fig.add_trace(data_trace(df["PHI"], vp))
    phi_full = np.linspace(0.0, 0.45, 150)
    fig.add_trace(go.Scatter(x=phi_full, y=wyllie(phi_full, v_matrix, v_fluid),
                             mode="lines", name="Wyllie time-average",
                             line=dict(color="#d62728", width=2)))
    fig.add_trace(go.Scatter(x=phi_full, y=raymer_hunt_gardner(phi_full, v_matrix, v_fluid),
                             mode="lines", name="Raymer-Hunt-Gardner",
                             line=dict(color="#ff7f0e", width=2)))

    # Saturated bounds: the mineral-fluid mixture, no substitution involved.
    hs = hashin_shtrikman(K_min, G_min, K_fl, 0.0, 1.0 - phi_full)
    rho_sat = (1.0 - phi_full) * rho_min + phi_full * rho_fl
    vp_hs_up, _ = velocities_from_moduli(hs["K_upper"], hs["G_upper"], rho_sat)
    vp_hs_lo, _ = velocities_from_moduli(hs["K_lower"], hs["G_lower"], rho_sat)
    fig.add_trace(go.Scatter(x=phi_full, y=vp_hs_up, mode="lines",
                             name="Hashin-Shtrikman upper",
                             line=dict(color="#1f77b4", width=1.6, dash="dash")))
    fig.add_trace(go.Scatter(x=phi_full, y=vp_hs_lo, mode="lines",
                             name="Hashin-Shtrikman lower (suspension)",
                             line=dict(color="#1f77b4", width=1.6, dash="dot")))
    st.plotly_chart(finish(fig, "Porosity (v/v)", "Vp (m/s)"), use_container_width=True)
    c1, c2 = st.columns(2)
    c1.metric("Matrix Vp", f"{v_matrix:.0f} m/s")
    c2.metric(f"{fluid_name.title()} Vp", f"{v_fluid:.0f} m/s")
    st.caption(
        "The Hashin-Shtrikman pair are bounds on the **saturated** mineral-plus-fluid "
        "mixture, computed directly — no Gassmann step is involved. Data must fall "
        "between them; where it falls says how the pore space is arranged. A point "
        "**below** the lower (suspension) bound cannot be that mineral with that fluid "
        "at that porosity — which is the diagnostic working: a gas sand drops below a "
        "brine bound and comes back inside once the pore fluid is switched to gas."
    )

# ------------------------------------------------------ moduli & bounds -----
with tab_moduli:
    K_log = bulk_modulus(vp, vs, rho)
    G_log = shear_modulus(vs, rho)

    hs = hashin_shtrikman(K_min, G_min, K_fl, 0.0, 1.0 - phi_full)
    K_soft, G_soft = soft_sand_dry(K_min, G_min, phi_grid, phi_c, n_grains,
                                   pressure, shear_factor)
    K_stiff, G_stiff = stiff_sand_dry(K_min, G_min, phi_grid, phi_c, n_grains,
                                      pressure, shear_factor)
    K_crit, G_crit = critical_porosity_dry(K_min, G_min, phi_grid, phi_c)

    which = st.radio("Modulus", ["Bulk K", "Shear μ"], horizontal=True)
    if which == "Bulk K":
        log_values, sat_up, sat_lo = K_log, hs["K_upper"], hs["K_lower"]
        dry_curves = [("Soft sand (dry)", K_soft), ("Stiff sand (dry)", K_stiff),
                      ("Critical porosity (dry)", K_crit)]
        axis = "K (GPa)"
    else:
        log_values, sat_up, sat_lo = G_log, hs["G_upper"], hs["G_lower"]
        dry_curves = [("Soft sand (dry)", G_soft), ("Stiff sand (dry)", G_stiff),
                      ("Critical porosity (dry)", G_crit)]
        axis = "μ (GPa)"

    fig = go.Figure()
    if has_phi:
        fig.add_trace(data_trace(df["PHI"], log_values, name="Well (saturated)"))
    fig.add_trace(go.Scatter(x=phi_full, y=sat_up, mode="lines",
                             name="Hashin-Shtrikman upper (saturated)",
                             line=dict(color="#1f77b4", width=2)))
    fig.add_trace(go.Scatter(x=phi_full, y=sat_lo, mode="lines",
                             name="Hashin-Shtrikman lower (saturated)",
                             line=dict(color="#1f77b4", width=2, dash="dot")))
    for name, curve in dry_curves:
        fig.add_trace(go.Scatter(x=phi_grid, y=curve, mode="lines", name=name,
                                 line=dict(width=1.8, dash="dash")))
    st.plotly_chart(finish(fig, "Porosity (v/v)", axis, height=600),
                    use_container_width=True)
    st.warning(
        "The three frame models are **dry-frame** curves. Comparing them directly "
        "against saturated log moduli is only valid for the shear modulus, which "
        "fluid does not change. Bringing the dry K curves up to saturated K needs "
        "Gassmann, which this toolkit deliberately does not carry.",
        icon=":material/warning:",
    )

    fig2 = go.Figure()
    fig2.add_trace(data_trace(K_log, G_log))
    lim = float(np.nanmax(K_log)) * 1.1
    fig2.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines", name="μ = K",
                              line=dict(color="#999", width=1, dash="dot")))
    st.plotly_chart(finish(fig2, "Bulk modulus K (GPa)", "Shear modulus μ (GPa)",
                           height=480), use_container_width=True)
