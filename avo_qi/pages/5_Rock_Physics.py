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
    wyllie,
)
from avo_qi.core.mixing import (  # noqa: E402
    MINERAL_MIXING_LAWS,
    fluid_density,
    fluid_mix,
    mineral_mix,
)
from avo_qi.ui import (  # noqa: E402
    add_derived_curves,
    apply_zone_filter,
    zone_labels,
    apply_lithology_filter,
    lithology_colour,
    lithology_labels,
    page_setup,
    require_well,
    sidebar,
)

page_setup("Rock Physics", icon=":rock:")
settings = sidebar(show_wavelet=False, show_angles=False, show_classifier=False)
well = require_well()

df = add_derived_curves(well.complete(settings.case).reset_index(drop=True))
litho = lithology_labels(df, settings)
zones = zone_labels(df, well, settings)
keep = apply_lithology_filter(df, litho, settings) & apply_zone_filter(zones, settings)
df = df[keep].reset_index(drop=True)
litho = litho[keep]
zones = zones[keep]
if df.empty:
    st.warning("The lithology filter has excluded every sample. Widen it in the sidebar.")
    st.stop()
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
mineral_tab, fluid_tab, frame_tab = st.tabs(
    ["Mineral matrix", "Pore fluid", "Frame model"]
)

with mineral_tab:
    st.caption(
        "Bounds are the honest answer for a mixture — the geometry that would "
        "pin down a single value is not known. Pick which bound, or which "
        "average of them, to carry into the models."
    )
    mineral_names = list(MINERALS)
    chosen = st.multiselect("Minerals", mineral_names, default=["quartz", "clay"])
    if not chosen:
        st.warning("Pick at least one mineral.")
        st.stop()
    cols = st.columns(max(len(chosen), 1))
    raw_fractions = [
        cols[i].number_input(f"{name} fraction", 0.0, 1.0,
                             1.0 if i == 0 else 0.0, 0.05, key=f"minfrac_{name}")
        for i, name in enumerate(chosen)
    ]
    if sum(raw_fractions) <= 0:
        st.warning("Mineral fractions must sum to more than zero.")
        st.stop()
    mineral_law = st.selectbox(
        "Mixing law", MINERAL_MIXING_LAWS, index=MINERAL_MIXING_LAWS.index("hill"),
        format_func=lambda m: {
            "voigt": "Voigt (upper bound)", "reuss": "Reuss (lower bound)",
            "hill": "Voigt-Reuss-Hill average",
            "hs_upper": "Hashin-Shtrikman upper", "hs_lower": "Hashin-Shtrikman lower",
            "hs_average": "Hashin-Shtrikman average",
        }[m],
    )
    K_min, G_min, rho_min = mineral_mix(
        [MINERALS[n][0] for n in chosen], [MINERALS[n][1] for n in chosen],
        [MINERALS[n][2] for n in chosen], raw_fractions, method=mineral_law,
    )
    total = sum(raw_fractions)
    st.caption("Normalised: " + ", ".join(
        f"{n} {f / total:.0%}" for n, f in zip(chosen, raw_fractions)))

with fluid_tab:
    st.caption(
        "How the phases are arranged matters as much as how much of each there "
        "is. Finely mixed, they share a pressure and average harmonically "
        "(Wood) — a little gas dominates. Segregated into patches, they stiffen "
        "independently and average arithmetically. Brie parks a saturation "
        "between the two."
    )
    fluid_names = list(FLUIDS)
    case_fluid = settings.case if settings.case in fluid_names else "brine"
    fluid_law = st.selectbox(
        "Fluid mixing law", ["single", "wood", "brie", "hill", "patchy"],
        format_func=lambda m: {
            "single": "Single phase", "wood": "Wood (uniform saturation)",
            "brie": "Brie (empirical)", "hill": "Hill (midpoint)",
            "patchy": "Patchy (segregated)",
        }[m],
    )
    if fluid_law == "single":
        fluid_name = st.selectbox("Pore fluid", fluid_names,
                                  index=fluid_names.index(case_fluid),
                                  help="Defaults to the case chosen in the sidebar.")
        K_fl, rho_fl = FLUIDS[fluid_name]
        mix_label = fluid_name
    else:
        c1, c2, c3 = st.columns(3)
        sw = c1.slider("Sw (brine)", 0.0, 1.0, 0.7, 0.05)
        so = c2.slider("So (oil)", 0.0, 1.0, 0.0, 0.05)
        sg = c3.slider("Sg (gas)", 0.0, 1.0, 0.3, 0.05)
        if sw + so + sg <= 0:
            st.warning("Saturations must sum to more than zero.")
            st.stop()
        brie_exponent = st.slider("Brie exponent", 1.0, 8.0, 3.0, 0.5,
                                  disabled=fluid_law != "brie",
                                  help="1 reproduces the patchy limit; larger "
                                       "values bend towards Wood.")
        moduli = [FLUIDS["brine"][0], FLUIDS["oil"][0], FLUIDS["gas"][0]]
        densities = [FLUIDS["brine"][1], FLUIDS["oil"][1], FLUIDS["gas"][1]]
        saturations = [sw, so, sg]
        K_fl = fluid_mix(moduli, saturations, fluid_law, brie_exponent=brie_exponent,
                         gas_index=2)
        rho_fl = fluid_density(densities, saturations)
        total_s = sw + so + sg
        mix_label = (f"{fluid_law} — Sw {sw / total_s:.0%} / So {so / total_s:.0%}"
                     f" / Sg {sg / total_s:.0%}")

        span = {name: fluid_mix(moduli, saturations, name, brie_exponent=brie_exponent,
                                gas_index=2)
                for name in ("wood", "brie", "hill", "patchy")}
        st.caption("Same saturations under every law: " + " · ".join(
            f"**{k}** {v:.3f}" for k, v in span.items()) + " GPa")

with frame_tab:
    c1, c2 = st.columns(2)
    phi_c = c1.slider("Critical porosity φc", 0.20, 0.50, 0.36, 0.01)
    pressure_mpa = c1.slider("Effective pressure (MPa)", 1.0, 60.0, 10.0, 1.0)
    n_default = float(coordination_number(phi_c))
    n_grains = c2.slider("Coordination number n", 4.0, 16.0, round(n_default, 1), 0.1,
                         help=f"Murphy's relation gives {n_default:.1f} at φc = {phi_c:.2f}.")
    shear_factor = c2.slider("Shear factor f", 0.0, 1.0, 1.0, 0.05,
                             help="1 = perfect adhesion between grains, 0 = frictionless.")

fluid_name = mix_label
pressure = pressure_mpa * 1e6

c1, c2, c3, c4 = st.columns(4)
c1.metric("Mineral K", f"{K_min:.1f} GPa")
c2.metric("Mineral μ", f"{G_min:.1f} GPa")
c3.metric("Mineral ρ", f"{rho_min:.2f} g/cc")
K_hm, G_hm = hertz_mindlin(K_min, G_min, phi_c, n_grains, pressure, shear_factor)
c4.metric("Hertz-Mindlin K (dry)", f"{K_hm:.2f} GPa")

phi_grid = np.linspace(0.0, phi_c, 120)
colour_options = ["LITHOLOGY"] + [c for c in ("DEPTH", "GR", "VSH", "PHI", "SW", "VPVS")
                                  if c in df.columns]
colour = st.selectbox("Colour the well data by", colour_options, index=0)


def data_trace(x, y, name="Well data"):
    """Well data as one trace, or one per lithology when coloured by it."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if colour != "LITHOLOGY":
        return [go.Scatter(
            x=x, y=y, mode="markers", name=name,
            marker=dict(size=5, opacity=0.75, color=df[colour], colorscale="Viridis",
                        showscale=True, colorbar=dict(title=colour, x=1.02)),
            hovertemplate="%{x:.4g}, %{y:.4g}<extra></extra>",
        )]
    traces = []
    for label in dict.fromkeys(litho):
        mask = litho == label
        traces.append(go.Scatter(
            x=x[mask], y=y[mask], mode="markers", name=label,
            marker=dict(size=5, opacity=0.85, color=lithology_colour(label),
                        line=dict(width=0.3, color="#555")),
            hovertemplate=f"%{{x:.4g}}, %{{y:.4g}}<extra>{label}</extra>",
        ))
    return traces


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
    fig.add_traces(data_trace(vp, vs))
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
    fig2.add_traces(data_trace(vp, df["VPVS"]))
    fig2.add_trace(go.Scatter(x=grid, y=grid / castagna_mudrock(grid), mode="lines",
                              name="Mudrock line", line=dict(color="#d62728", width=2.5)))
    st.plotly_chart(finish(fig2, "Vp (m/s)", "Vp/Vs", height=460), use_container_width=True)

# ------------------------------------------------------------- Gardner ------
with tab_gardner:
    a_fit, b_fit = fit_gardner(vp, rho)
    fig = go.Figure()
    fig.add_traces(data_trace(vp, rho))
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
        fig.add_traces(data_trace(df["PHI"], vp))
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
        fig.add_traces(data_trace(df["PHI"], log_values, name="Well (saturated)"))
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
    fig2.add_traces(data_trace(K_log, G_log))
    lim = float(np.nanmax(K_log)) * 1.1
    fig2.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines", name="μ = K",
                              line=dict(color="#999", width=1, dash="dot")))
    st.plotly_chart(finish(fig2, "Bulk modulus K (GPa)", "Shear modulus μ (GPa)",
                           height=480), use_container_width=True)
