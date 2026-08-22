"""Page 5 — rock physics: bounds, trends, frame models and a forward model.

The overlays here are *models* to diagnose the well against, and the page keeps
them independent of it.  Deriving a model's inputs from the measurements it is
then tested against would guarantee agreement and destroy the diagnostic; the
value of a model is that it can disagree.

The **Forward model** tab is the exception that proves the rule.  It is driven
from ``VSH``, ``PHIT`` and ``SW``, which come from gamma ray, density and
resistivity — not from Vp and Vs — so the velocities it predicts are a real
test of the model against logs it never saw.  ``PHIT`` is the one input that
can be circular, and the tab says so when it is.
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
from avo_qi.core.fluids import (  # noqa: E402
    fluid_properties,
    geothermal_temperature,
    hydrostatic_pressure,
)
from avo_qi.core.gassmann import gassmann_saturate  # noqa: E402
from avo_qi.core.misfit import bounds_check, log_misfit  # noqa: E402
from avo_qi.core.petro import (  # noqa: E402
    FRAME_MODELS,
    forward_model,
    porosity_provenance,
)
from avo_qi.io.loader import FLUID_CASES, add_substituted_case  # noqa: E402
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
    "Model overlays for diagnosing the well — mixture bounds, empirical trends, "
    "granular frame models, and a forward model driven from the petrophysical "
    "logs. The overlays are set by hand and stay independent of the elastic "
    "logs, so where the data misses them, that is a finding rather than a "
    "fitting error."
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

    if "VSH" in df.columns and np.isfinite(df["VSH"]).any():
        vsh_ref = df["VSH"].to_numpy(float)
        st.caption(
            f"For reference, this selection measures **VSH {np.nanmedian(vsh_ref):.2f}** "
            f"(P10–P90 {np.nanpercentile(vsh_ref, 10):.2f}–"
            f"{np.nanpercentile(vsh_ref, 90):.2f}). Shown, not applied: the gap "
            "between the matrix you set and what the well measures is itself "
            "information, and defaulting the controls to the well would hide it. "
            "The **Forward model** tab is where the logs drive the model."
        )

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
    brie_exponent = 3.0          # overridden below when the Brie law is chosen

    depth_sel = df["DEPTH"].to_numpy(float)
    with st.expander("Reservoir conditions — Batzle-Wang", expanded=False):
        st.caption(
            "The built-in table is a set of room-condition constants. At 30 MPa "
            "and 90 °C a gas is roughly five times stiffer and six times denser "
            "than its entry, so a bound — or a substitution — built on the table "
            "is confidently wrong rather than approximately right. Switch this on "
            "to compute K and ρ from pressure and temperature instead."
        )
        use_bw = st.checkbox("Compute fluid properties from pressure and temperature",
                             value=False, key="bw_enabled")
        b1, b2, b3 = st.columns(3)
        p_gradient = b1.slider("Pressure gradient (MPa/km)", 8.0, 20.0, 10.0, 0.5,
                               disabled=not use_bw,
                               help="10 is a normally pressured brine column; "
                                    "raise it for overpressure.")
        t_surface = b2.slider("Surface temperature (°C)", 0.0, 40.0, 15.0, 1.0,
                              disabled=not use_bw)
        t_gradient = b3.slider("Geothermal gradient (°C/km)", 15.0, 50.0, 30.0, 1.0,
                               disabled=not use_bw)
        b4, b5, b6 = st.columns(3)
        salinity_ppk = b4.slider("Salinity (ppm ×1000)", 0.0, 250.0, 35.0, 5.0,
                                 disabled=not use_bw,
                                 help="35 is seawater; formation brines run far higher.")
        oil_api = b5.slider("Oil gravity (°API)", 10.0, 55.0, 30.0, 1.0,
                            disabled=not use_bw)
        oil_gor = b6.slider("GOR (L/L)", 0.0, 300.0, 0.0, 10.0, disabled=not use_bw,
                            help="Dissolved gas softens oil sharply — 200 L/L "
                                 "roughly halves its modulus.")
        gas_gravity = st.slider("Gas gravity (relative to air)", 0.55, 1.20, 0.65,
                                0.01, disabled=not use_bw)

    salinity = salinity_ppk / 1000.0
    p_sample = hydrostatic_pressure(depth_sel, p_gradient)
    t_sample = geothermal_temperature(depth_sel, t_surface, t_gradient)
    p_mid = float(np.nanmedian(p_sample)) if p_sample.size else 0.0
    t_mid = float(np.nanmedian(t_sample)) if t_sample.size else 0.0

    def fluid_constants(name, per_sample=False):
        """``(K, rho)`` for one fluid — the fixed table, or Batzle-Wang.

        ``per_sample`` returns arrays following the depth profile, which is what
        the forward model and a substitution want; the bound curves are drawn
        once and take the mid-depth values.
        """
        if not use_bw:
            return FLUIDS[name]
        pressure_in = p_sample if per_sample else p_mid
        temperature_in = t_sample if per_sample else t_mid
        k, rho, _ = fluid_properties(name, pressure_in, temperature_in,
                                     salinity=salinity, api=oil_api, gor=oil_gor,
                                     gas_gravity=gas_gravity)
        return k, rho

    if use_bw:
        _, _, bw_warning = fluid_properties("brine", p_sample, t_sample,
                                            salinity=salinity)
        st.caption(
            f"At the selection's mid-depth ({float(np.nanmedian(depth_sel)):,.0f} m "
            f"→ {p_mid:.1f} MPa, {t_mid:.0f} °C): " + " · ".join(
                f"**{n}** {fluid_constants(n)[0]:.3f} GPa / "
                f"{fluid_constants(n)[1]:.3f} g/cc" for n in fluid_names)
            + ". Table values: " + " · ".join(
                f"{n} {FLUIDS[n][0]:.3f} / {FLUIDS[n][1]:.2f}" for n in fluid_names))
        if bw_warning:
            st.warning(bw_warning, icon=":material/warning:")

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
        K_fl, rho_fl = fluid_constants(fluid_name)
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
        moduli = [float(fluid_constants(n)[0]) for n in ("brine", "oil", "gas")]
        densities = [float(fluid_constants(n)[1]) for n in ("brine", "oil", "gas")]
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


tab_vpvs, tab_gardner, tab_vphi, tab_moduli, tab_forward, tab_subs = st.tabs(
    ["Vp – Vs trends", "Gardner", "Velocity – porosity", "Moduli & bounds",
     "Forward model", "Fluid substitution"]
)


def misfit_note(check, unit, depths):
    """Show a bounds check as a coloured one-liner, sized by how bad it is."""
    text = check.describe(unit=unit, depths=depths)
    if check.n == 0:
        st.caption(text)
    elif check.fraction_inside >= 0.99:
        st.success(text, icon=":material/check_circle:")
    elif check.fraction_inside >= 0.90:
        st.info(text, icon=":material/info:")
    else:
        st.warning(text, icon=":material/warning:")

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

    vphi_check = None
    if has_phi:
        phi_data = df["PHI"].to_numpy(float)
        lo_at = np.interp(phi_data, phi_full, vp_hs_lo, left=np.nan, right=np.nan)
        hi_at = np.interp(phi_data, phi_full, vp_hs_up, left=np.nan, right=np.nan)
        vphi_check = bounds_check(vp, lo_at, hi_at)
        if st.checkbox("Ring the samples that fall outside the bounds",
                       value=True, key="vphi_ring") and vphi_check.outside.size:
            out = vphi_check.outside
            fig.add_trace(go.Scatter(
                x=phi_data[out], y=vp[out], mode="markers", name="Outside the bounds",
                marker=dict(size=9, color="rgba(0,0,0,0)",
                            line=dict(width=1.8, color="#d62728")),
                hovertemplate="φ %{x:.3f}, Vp %{y:.0f}<extra>outside</extra>"))

    st.plotly_chart(finish(fig, "Porosity (v/v)", "Vp (m/s)"), use_container_width=True)
    if vphi_check is not None:
        misfit_note(vphi_check, "m/s", df["DEPTH"].to_numpy(float))
    c1, c2 = st.columns(2)
    c1.metric("Matrix Vp", f"{v_matrix:.0f} m/s")
    c2.metric(f"{fluid_name.title()} Vp", f"{v_fluid:.0f} m/s")
    st.caption(
        "The Hashin-Shtrikman pair are bounds on the **saturated** mineral-plus-fluid "
        "mixture, computed directly from the mixture rather than by substituting the "
        "well. Data must fall between them; where it falls says how the pore space is "
        "arranged. A point **below** the lower (suspension) bound cannot be that "
        "mineral with that fluid at that porosity — which is the diagnostic working: a "
        "gas sand drops below a brine bound and comes back inside once the pore fluid "
        "is switched to gas. The percentage above is that test counted rather than "
        "eyeballed."
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
    saturate = st.checkbox(
        "Bring the dry frame curves up to saturated with Gassmann", value=True,
        help="The log moduli are saturated, so a dry curve is not a like-for-like "
             "comparison for K. Gassmann fixes that; μ needs no correction.",
    )

    if which == "Bulk K":
        log_values, sat_up, sat_lo = K_log, hs["K_upper"], hs["K_lower"]
        frame_curves = [("Soft sand", K_soft), ("Stiff sand", K_stiff),
                        ("Critical porosity", K_crit)]
        axis = "K (GPa)"
        if saturate:
            frame_curves = [
                (name, gassmann_saturate(curve, K_min, K_fl, phi_grid).values)
                for name, curve in frame_curves
            ]
        suffix = " (saturated)" if saturate else " (dry)"
    else:
        log_values, sat_up, sat_lo = G_log, hs["G_upper"], hs["G_lower"]
        # Fluid does not touch the shear modulus, so dry is already saturated.
        frame_curves = [("Soft sand", G_soft), ("Stiff sand", G_stiff),
                        ("Critical porosity", G_crit)]
        axis = "μ (GPa)"
        suffix = " (dry = saturated)"

    fig = go.Figure()
    if has_phi:
        fig.add_traces(data_trace(df["PHI"], log_values, name="Well (saturated)"))
    fig.add_trace(go.Scatter(x=phi_full, y=sat_up, mode="lines",
                             name="Hashin-Shtrikman upper (saturated)",
                             line=dict(color="#1f77b4", width=2)))
    fig.add_trace(go.Scatter(x=phi_full, y=sat_lo, mode="lines",
                             name="Hashin-Shtrikman lower (saturated)",
                             line=dict(color="#1f77b4", width=2, dash="dot")))
    for name, curve in frame_curves:
        fig.add_trace(go.Scatter(x=phi_grid, y=curve, mode="lines", name=name + suffix,
                                 line=dict(width=1.8, dash="dash")))

    moduli_check = None
    if has_phi:
        phi_data = df["PHI"].to_numpy(float)
        lo_at = np.interp(phi_data, phi_full, sat_lo, left=np.nan, right=np.nan)
        hi_at = np.interp(phi_data, phi_full, sat_up, left=np.nan, right=np.nan)
        moduli_check = bounds_check(log_values, lo_at, hi_at)
        if moduli_check.outside.size:
            out = moduli_check.outside
            fig.add_trace(go.Scatter(
                x=phi_data[out], y=np.asarray(log_values)[out], mode="markers",
                name="Outside the bounds",
                marker=dict(size=9, color="rgba(0,0,0,0)",
                            line=dict(width=1.8, color="#d62728")),
                hovertemplate="φ %{x:.3f}, %{y:.3f}<extra>outside</extra>"))

    st.plotly_chart(finish(fig, "Porosity (v/v)", axis, height=600),
                    use_container_width=True)
    if moduli_check is not None:
        misfit_note(moduli_check, "GPa", df["DEPTH"].to_numpy(float))
    if saturate and which == "Bulk K":
        st.caption(
            "The frame curves have been carried from dry to saturated through "
            "Gassmann with the pore fluid set on the **Pore fluid** tab, so they "
            "are comparable with the log moduli directly. Untick the box to see "
            "the dry frames the models actually produce."
        )
    elif which == "Bulk K":
        st.warning(
            "These are **dry-frame** curves and the log moduli are saturated, so "
            "they are not a like-for-like comparison. Tick the box above to bring "
            "them up with Gassmann.",
            icon=":material/warning:",
        )
    else:
        st.caption(
            "Shear needs no Gassmann step: a fluid supports no shear, so the dry "
            "and saturated shear moduli are the same curve. This is the one "
            "modulus a dry frame model can be checked against directly."
        )

    fig2 = go.Figure()
    fig2.add_traces(data_trace(K_log, G_log))
    lim = float(np.nanmax(K_log)) * 1.1
    fig2.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines", name="μ = K",
                              line=dict(color="#999", width=1, dash="dot")))
    st.plotly_chart(finish(fig2, "Bulk modulus K (GPa)", "Shear modulus μ (GPa)",
                           height=480), use_container_width=True)


# ------------------------------------------------------- forward model ------
with tab_forward:
    st.caption(
        "A model built sample by sample from the petrophysics: VSH sets the "
        "matrix, SW sets the pore fluid, PHIT sets the porosity, a granular "
        "frame model and Gassmann do the rest. It predicts **Vp, Vs and RHOB "
        "logs**, which are then compared with the measured ones. VSH comes from "
        "gamma ray and SW from resistivity, so neither has seen a velocity — "
        "the comparison is a real test, not a fit."
    )

    if not has_phi:
        st.info(
            "The forward model needs a porosity curve. Map a PHI curve on the "
            "**Load & QC** page to use this tab.", icon=":material/info:")
    else:
        f1, f2, f3 = st.columns(3)
        frame_choice = f1.selectbox("Frame model", sorted(FRAME_MODELS),
                                    index=sorted(FRAME_MODELS).index("soft sand"))
        shale_mineral = f2.selectbox(
            "Shale mineral", list(MINERALS), index=list(MINERALS).index("clay"),
            help="The mineral that VSH is taken to be made of.")
        hc_options = [n for n in FLUIDS if n != "brine"]
        hydrocarbon = f3.selectbox(
            "Hydrocarbon", hc_options,
            index=hc_options.index("gas") if "gas" in hc_options else 0,
            help="What fills the pore space that SW says is not water.")

        # The matrix is whatever the Mineral matrix tab chose, minus whichever
        # mineral is standing in for the shale — VSH supplies that fraction.
        matrix_blend = {n: f for n, f in zip(chosen, raw_fractions)
                        if n != shale_mineral and f > 0}
        if not matrix_blend:
            matrix_blend = {"quartz": 1.0}
            st.caption("No non-shale mineral is selected, so the matrix falls "
                       "back to pure quartz.")

        g1, g2 = st.columns(2)
        if "VSH" in df.columns and np.isfinite(df["VSH"]).any():
            vsh_log = df["VSH"].to_numpy(float)
            g1.caption(f"**VSH** from the well — median {np.nanmedian(vsh_log):.2f} "
                       f"(P10–P90 {np.nanpercentile(vsh_log, 10):.2f}–"
                       f"{np.nanpercentile(vsh_log, 90):.2f})")
        else:
            vsh_log = np.full(len(df), g1.slider("VSH (assumed)", 0.0, 1.0, 0.1, 0.05))
            g1.caption("This well has no VSH curve, so a constant is assumed.")
        if "SW" in df.columns and np.isfinite(df["SW"]).any():
            sw_log = df["SW"].to_numpy(float)
            g2.caption(f"**SW** from the well — median {np.nanmedian(sw_log):.2f} "
                       f"(P10–P90 {np.nanpercentile(sw_log, 10):.2f}–"
                       f"{np.nanpercentile(sw_log, 90):.2f})")
        else:
            sw_log = np.full(len(df), g2.slider("SW (assumed)", 0.0, 1.0, 1.0, 0.05))
            g2.caption("This well has no SW curve, so a constant is assumed.")

        phi_log = df["PHI"].to_numpy(float)
        depth_log = df["DEPTH"].to_numpy(float)

        model = forward_model(
            vsh_log, phi_log, sw_log,
            matrix=matrix_blend, shale=shale_mineral, mineral_law=mineral_law,
            frame=frame_choice,
            # "Single phase" is a choice about the bound curves, not about how
            # the model should mix a partial saturation; Wood is the default there.
            fluid_law="wood" if fluid_law == "single" else fluid_law,
            brie_exponent=brie_exponent,
            hydrocarbon=hydrocarbon,
            brine=fluid_constants("brine", per_sample=True),
            hydrocarbon_properties=fluid_constants(hydrocarbon, per_sample=True),
            phi_c=phi_c, n_grains=n_grains, pressure=pressure,
            shear_factor=shear_factor,
        )

        provenance = porosity_provenance(phi_log, rho,
                                         mnemonic=well.mapping.get("PHI"))

        # --- how far the model sits from the well -----------------------------
        st.subheader("Misfit")
        measured = {"VP": vp, "VS": vs, "RHOB": rho}
        units = {"VP": "m/s", "VS": "m/s", "RHOB": "g/cc"}
        cols = st.columns(3)
        for col, curve in zip(cols, ("VP", "VS", "RHOB")):
            fit = log_misfit(model[curve], measured[curve])
            circular = curve == "RHOB" and not provenance.rhob_check_is_meaningful
            with col:
                if fit.n == 0:
                    st.caption(fit.describe(name=curve))
                    continue
                st.metric(
                    f"{curve} bias" + (" (circular)" if circular else ""),
                    f"{fit.bias:+.4g} {units[curve]}",
                    f"{fit.relative_bias * 100:+.1f}%",
                    # Neither sign is the good one — a misfit is bad in both
                    # directions — so the arrow is left uncoloured rather than
                    # implying that running low beats running high.
                    delta_color="off",
                )
                st.caption(
                    ("_Not a test — see below._ " if circular else "")
                    + f"scatter {fit.scatter:.4g} {units[curve]}, "
                    + (f"correlation {fit.correlation:.2f}"
                       if np.isfinite(fit.correlation) else "correlation n/a"))

        if provenance.density_derived is True:
            st.warning(provenance.note, icon=":material/warning:")
        else:
            st.caption(provenance.note)

        bad = int((~model["valid"]).sum())
        if bad:
            from collections import Counter
            counts = Counter(str(r) for r in model["reasons"][~model["valid"]])
            st.info(
                f"{bad:,} of {len(phi_log):,} samples have no prediction: "
                + "; ".join(f"{reason} ({n:,})" for reason, n in counts.most_common()),
                icon=":material/info:")

        # --- the logs, measured against modelled ------------------------------
        from plotly.subplots import make_subplots

        tracks = make_subplots(rows=1, cols=3, shared_yaxes=True,
                               subplot_titles=("Vp (m/s)", "Vs (m/s)", "RHOB (g/cc)"),
                               horizontal_spacing=0.04)
        for i, curve in enumerate(("VP", "VS", "RHOB"), start=1):
            tracks.add_trace(go.Scatter(
                x=measured[curve], y=depth_log, mode="lines", name="Measured",
                legendgroup="measured", showlegend=(i == 1),
                line=dict(color="#333", width=1.4)), row=1, col=i)
            tracks.add_trace(go.Scatter(
                x=model[curve], y=depth_log, mode="lines", name="Model",
                legendgroup="model", showlegend=(i == 1),
                line=dict(color="#d62728", width=1.6, dash="dot")), row=1, col=i)
        tracks.update_yaxes(autorange="reversed", title_text="Depth (m)", row=1, col=1)
        tracks.update_layout(height=760, margin=dict(l=60, r=20, t=60, b=40),
                             legend=dict(orientation="h", yanchor="bottom", y=1.04))
        st.plotly_chart(tracks, use_container_width=True)

        cross = go.Figure()
        cross.add_traces(data_trace(vp, vs, name="Measured"))
        cross.add_trace(go.Scatter(
            x=model["VP"], y=model["VS"], mode="markers", name="Model",
            marker=dict(size=5, opacity=0.6, color="#d62728", symbol="x")))
        st.plotly_chart(finish(cross, "Vp (m/s)", "Vs (m/s)", height=520),
                        use_container_width=True)
        st.caption(
            "Where the model cloud sits apart from the data, the disagreement is "
            "the result. A systematic bias usually means the frame model or the "
            "effective pressure is wrong for this rock; scatter that grows with "
            "porosity usually means the matrix is."
        )


# ---------------------------------------------------- fluid substitution ----
with tab_subs:
    st.caption(
        "Gassmann-substitute the well into another pore fluid. The result is "
        "written as an ordinary fluid case, so it appears in the sidebar's case "
        "selector and flows through every other page — the synthetic gather, the "
        "AVO classification, the crossplots — exactly as a case loaded from the "
        "LAS would."
    )

    if not has_phi:
        st.info("Fluid substitution needs a porosity curve. Map a PHI curve on "
                "the **Load & QC** page to use this tab.", icon=":material/info:")
    else:
        s1, s2, s3 = st.columns(3)
        source_case = s1.selectbox("Substitute from", well.cases,
                                   index=well.cases.index(well.active_case))
        in_fluid = s2.selectbox("Fluid in the rock now", fluid_names,
                                index=fluid_names.index("brine"))
        out_fluid = s3.selectbox("Fluid to put in", fluid_names,
                                 index=fluid_names.index("gas"))
        storable = [c for c in FLUID_CASES if c != "in situ"]
        target_case = st.selectbox(
            "Store the result as", storable,
            index=storable.index(out_fluid) if out_fluid in storable else 0,
            help="Which case name the substituted curves take.")

        k_in, rho_in = fluid_constants(in_fluid, per_sample=True)
        k_out, rho_out = fluid_constants(out_fluid, per_sample=True)
        st.caption(
            f"Using {in_fluid} at {np.mean(np.atleast_1d(k_in)):.3f} GPa / "
            f"{np.mean(np.atleast_1d(rho_in)):.3f} g/cc and {out_fluid} at "
            f"{np.mean(np.atleast_1d(k_out)):.3f} GPa / "
            f"{np.mean(np.atleast_1d(rho_out)):.3f} g/cc, with a mineral K of "
            f"{K_min:.1f} GPa from the **Mineral matrix** tab."
            + ("" if use_bw else " Switch on Batzle-Wang under **Pore fluid** to "
                                "use reservoir conditions instead of the table.")
        )

        # Storing over a case that came out of the LAS would throw away measured
        # curves, which is never something to do quietly.
        overwrites_loaded = (target_case in well.cases
                             and not well.is_computed(target_case))
        if overwrites_loaded:
            st.warning(
                f"This well already carries a **{target_case}** case that was "
                "loaded from the file. Substituting will replace those curves "
                "with modelled ones for the rest of this session. Store the "
                "result under a different case if you want to keep both.",
                icon=":material/warning:")

        if st.button(f"Substitute {in_fluid} → {out_fluid}", type="primary"):
            # The whole well, not the filtered selection: a case has to line up
            # with every other curve for the other pages to use it.
            full = well.frame(source_case)
            phi_full_well = full["PHI"].to_numpy(float)
            depth_full = full["DEPTH"].to_numpy(float)
            if use_bw:
                p_full = hydrostatic_pressure(depth_full, p_gradient)
                t_full = geothermal_temperature(depth_full, t_surface, t_gradient)
                fluid_in_props = fluid_properties(
                    in_fluid, p_full, t_full, salinity=salinity, api=oil_api,
                    gor=oil_gor, gas_gravity=gas_gravity)[:2]
                fluid_out_props = fluid_properties(
                    out_fluid, p_full, t_full, salinity=salinity, api=oil_api,
                    gor=oil_gor, gas_gravity=gas_gravity)[:2]
            else:
                fluid_in_props, fluid_out_props = FLUIDS[in_fluid], FLUIDS[out_fluid]

            result = add_substituted_case(
                well, target_case, fluid_in_props, fluid_out_props,
                phi_full_well, K_min, source_case=source_case)

            done = int(result["valid"].sum())
            total_n = int(result["valid"].size)
            st.success(
                f"Substituted {done:,} of {total_n:,} samples into the "
                f"**{target_case}** case. Pick it in the sidebar to carry it "
                "through the rest of the app.", icon=":material/check_circle:")
            if done < total_n:
                from collections import Counter
                counts = Counter(str(r) for r in result["reasons"][~result["valid"]])
                st.warning(
                    "Samples left empty: "
                    + "; ".join(f"{reason} ({n:,})" for reason, n in counts.most_common())
                    + ". A negative dry frame means the porosity, the mineral and "
                      "the measured velocities disagree — the rock cannot be what "
                      "those three say it is.",
                    icon=":material/warning:")

        if well.computed_cases:
            st.info(
                "Computed here, not loaded from the file: **"
                + "**, **".join(well.computed_cases)
                + "**. They are models of the well, not measurements of it.",
                icon=":material/functions:")
