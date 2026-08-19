"""Generate ``demo_well.las``: the three-layer shale / gas-sand / shale model.

The layer properties are the ones the physics contracts in SPEC.md section 4
are validated against — a soft Class III gas sand encased in shale.  Run from
the repository root::

    python -m avo_qi.sample_data.make_demo_well
"""

from __future__ import annotations

import os

import numpy as np

# Matrix and fluid densities (g/cc) used to derive porosity; see `density_porosity`.
RHO_CLAY, RHO_QUARTZ = 2.58, 2.65
RHO_BRINE, RHO_GAS = 1.09, 0.25


def density_porosity(rho, rho_matrix, rho_fluid):
    """Standard density porosity: phi = (rho_ma - rho_b) / (rho_ma - rho_fl).

    Deriving porosity this way keeps the demo well internally consistent —
    PHI and RHOB tell the same story — instead of carrying an invented
    porosity curve that the density contradicts.
    """
    return (rho_matrix - rho) / (rho_matrix - rho_fluid)


# (name, vp m/s, vs m/s, rho g/cc, gr API, vsh, rho_matrix, rho_fluid, sw)
#
# The elastic values for shale and gas sand are fixed by the validated AVO
# cross-check in SPEC.md section 4.1 and must not be changed.  They describe a
# shallow, poorly consolidated section, so they are softer than a consolidated
# quartz/brine rock physics model would predict at these porosities — some
# layers therefore plot below the suspension bound on the Rock Physics page,
# which is the diagnostic doing its job rather than a defect.
SHALE = ("shale", 2400.0, 1200.0, 2.35, 95.0, 0.85, RHO_CLAY, RHO_BRINE, 1.00)
GAS_SAND = ("gas sand", 2100.0, 1300.0, 2.10, 25.0, 0.10, RHO_QUARTZ, RHO_GAS, 0.20)
BRINE_SAND = ("brine sand", 2500.0, 1450.0, 2.30, 30.0, 0.12, RHO_QUARTZ, RHO_BRINE, 1.00)
HARD_STREAK = ("cemented sand", 3000.0, 1800.0, 2.45, 40.0, 0.15, RHO_QUARTZ, RHO_BRINE, 1.00)

#: (top_md, base_md, layer) — depths in metres.
LAYERS = [
    (2000.0, 2040.0, SHALE),
    (2040.0, 2070.0, GAS_SAND),
    (2070.0, 2090.0, SHALE),
    (2090.0, 2105.0, BRINE_SAND),
    (2105.0, 2120.0, SHALE),
    (2120.0, 2130.0, HARD_STREAK),
    (2130.0, 2160.0, SHALE),
]

STEP = 0.1524  # m, the usual half-foot LAS sampling
NOISE_SEED = 7
NOISE = {"VP": 8.0, "VS": 5.0, "RHOB": 0.006, "GR": 3.0}


def build_logs(step=STEP, layers=LAYERS, noise=True, seed=NOISE_SEED):
    """Return a dict of canonical logs for the layer-cake model."""
    top = layers[0][0]
    base = layers[-1][1]
    depth = np.arange(top, base + step / 2, step)

    curves = {k: np.full(depth.size, np.nan) for k in ("VP", "VS", "RHOB", "GR", "VSH", "PHI", "SW")}
    for z_top, z_base, layer in layers:
        _, vp, vs, rho, gr, vsh, rho_ma, rho_fl, sw = layer
        phi = density_porosity(rho, rho_ma, rho_fl)
        sel = (depth >= z_top) & (depth < z_base)
        for key, value in zip(curves, (vp, vs, rho, gr, vsh, phi, sw)):
            curves[key][sel] = value
    for key in curves:  # the final sample sits exactly on `base`
        curves[key] = np.where(np.isnan(curves[key]), curves[key][~np.isnan(curves[key])][-1], curves[key])

    if noise:
        rng = np.random.default_rng(seed)
        for key, sigma in NOISE.items():
            curves[key] = curves[key] + rng.normal(0.0, sigma, depth.size)

    curves["DEPTH"] = depth
    return curves


def write_las(path, curves):
    """Write the curves to a LAS 2.0 file."""
    import lasio

    las = lasio.LASFile()
    las.well["WELL"] = lasio.HeaderItem("WELL", value="DEMO-1")
    las.well["FLD"] = lasio.HeaderItem("FLD", value="AVO QI DEMO")
    las.well["COMP"] = lasio.HeaderItem("COMP", value="Rock Physics Toolkit")
    las.well["STRT"] = lasio.HeaderItem("STRT", unit="M", value=float(curves["DEPTH"][0]))
    las.well["STOP"] = lasio.HeaderItem("STOP", unit="M", value=float(curves["DEPTH"][-1]))
    las.well["STEP"] = lasio.HeaderItem("STEP", unit="M", value=float(STEP))
    las.well["NULL"] = lasio.HeaderItem("NULL", value=-999.25)

    units = {
        "DEPT": "M", "VP": "M/S", "VS": "M/S", "RHOB": "G/C3",
        "GR": "GAPI", "VSH": "V/V", "PHI": "V/V", "SW": "V/V",
    }
    descr = {
        "DEPT": "Measured depth", "VP": "Compressional velocity",
        "VS": "Shear velocity", "RHOB": "Bulk density",
        "GR": "Gamma ray", "VSH": "Shale volume",
        "PHI": "Effective porosity", "SW": "Water saturation",
    }

    las.append_curve("DEPT", curves["DEPTH"], unit=units["DEPT"], descr=descr["DEPT"])
    for key in ("VP", "VS", "RHOB", "GR", "VSH", "PHI", "SW"):
        las.append_curve(key, curves[key], unit=units[key], descr=descr[key])

    las.header["Other"] = (
        "Synthetic three-layer AVO demonstration well. Shale / Class III gas sand / "
        "shale, with a brine sand and a cemented hard streak below. Not real data."
    )
    las.write(path, version=2.0, fmt="%.4f")
    return path


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "demo_well.las")
    write_las(path, build_logs())
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
