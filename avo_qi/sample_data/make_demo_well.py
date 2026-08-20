"""Generate ``demo_well.las``: a three-layer well with fluid-substituted cases.

The well carries four fluid cases — in situ, brine, oil and gas — as the
suffixed curves ``VP_BR``, ``VS_OIL``, ``RHOB_GAS`` and so on, which is how a
well arrives once fluid substitution has been done upstream.

**The substitution was done offline, not by this toolkit.**  There is no
Gassmann and no Batzle-Wang in ``avo_qi/`` (SPEC.md section 1), so the
per-case elastic constants below are literals: they were computed once with
Gassmann from each layer's dry frame and are recorded here as data, exactly
as an upstream tool would have delivered them.

Layer provenance:

* **Shale** — 2400 / 1200 / 2.35 is fixed by the validated AVO cross-check in
  SPEC.md section 4.1 and must not be changed.  As a non-reservoir it carries
  the same values in every fluid case.
* **Gas sand** — the in-situ (gas) case 2100 / 1300 / 2.10 is fixed by the same
  cross-check.  Its brine and oil cases were Gassmann-substituted from the dry
  frame implied by those values.
* **Brine sand** and **cemented sand** — built from a Dvorkin-Nur dry frame
  (soft sand and stiff sand respectively) at 20 MPa effective pressure, then
  saturated.  Earlier hand-picked values for these two were not physically
  realisable: inverting Gassmann on them returned a negative dry-frame bulk
  modulus, and they plotted below the suspension bound on the Rock Physics
  page.  These are consistent.

Run from the repository root::

    python -m avo_qi.sample_data.make_demo_well
"""

from __future__ import annotations

import os

import numpy as np

# Matrix and fluid densities (g/cc) used to derive porosity.
RHO_CLAY, RHO_QUARTZ = 2.58, 2.65
RHO_BRINE, RHO_GAS, RHO_OIL = 1.09, 0.25, 0.78

#: Fluid cases written to the LAS, and the mnemonic suffix each one uses.
CASE_SUFFIX = {"brine": "BR", "oil": "OIL", "gas": "GAS"}


def density_porosity(rho, rho_matrix, rho_fluid):
    """Standard density porosity: phi = (rho_ma - rho_b) / (rho_ma - rho_fl).

    Deriving porosity this way keeps the demo well internally consistent —
    PHI and RHOB tell the same story — instead of carrying an invented
    porosity curve that the density contradicts.
    """
    return (rho_matrix - rho) / (rho_matrix - rho_fluid)


class Layer:
    """One layer: shared curves plus its (vp, vs, rho) triple per fluid case."""

    def __init__(self, name, gr, vsh, sw, rho_matrix, rho_fluid, cases, insitu):
        self.name = name
        self.gr = gr
        self.vsh = vsh
        self.sw = sw
        self.rho_matrix = rho_matrix
        self.rho_fluid = rho_fluid
        self.cases = dict(cases)
        self.cases["in situ"] = self.cases[insitu]
        self.insitu = insitu

    @property
    def phi(self):
        """Porosity from the in-situ density and the in-situ pore fluid."""
        return density_porosity(self.cases["in situ"][2], self.rho_matrix, self.rho_fluid)


#: A non-reservoir shale: fluid substitution does not move it.
SHALE = Layer(
    "shale", gr=95.0, vsh=0.85, sw=1.00,
    rho_matrix=RHO_CLAY, rho_fluid=RHO_BRINE, insitu="brine",
    cases={
        "brine": (2400.0, 1200.0, 2.3500),
        "oil": (2400.0, 1200.0, 2.3500),
        "gas": (2400.0, 1200.0, 2.3500),
    },
)

#: The Class III gas sand.  Its gas case is the spec-pinned one.
GAS_SAND = Layer(
    "gas sand", gr=25.0, vsh=0.10, sw=0.20,
    rho_matrix=RHO_QUARTZ, rho_fluid=RHO_GAS, insitu="gas",
    cases={
        "brine": (2717.9, 1244.2, 2.2925),
        "oil": (2337.6, 1264.0, 2.2215),
        "gas": (2100.0, 1300.0, 2.1000),
    },
)

#: A brine sand: soft-sand dry frame at 26% porosity, then saturated.
BRINE_SAND = Layer(
    "brine sand", gr=30.0, vsh=0.12, sw=1.00,
    rho_matrix=RHO_QUARTZ, rho_fluid=RHO_BRINE, insitu="brine",
    cases={
        "brine": (2830.2, 1498.9, 2.2444),
        "oil": (2496.8, 1526.6, 2.1638),
        "gas": (2315.8, 1577.6, 2.0260),
    },
)

#: A cemented streak: stiff-sand dry frame at 18% porosity.  A stiff frame is
#: barely fluid-sensitive, which is the point of including it.
HARD_STREAK = Layer(
    "cemented sand", gr=40.0, vsh=0.15, sw=1.00,
    rho_matrix=RHO_QUARTZ, rho_fluid=RHO_BRINE, insitu="brine",
    cases={
        "brine": (4265.3, 2704.1, 2.3692),
        "oil": (4176.4, 2736.5, 2.3134),
        "gas": (4180.6, 2794.8, 2.2180),
    },
)

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


def _curve_names():
    """Canonical curve names written to the LAS, in order."""
    names = ["VP", "VS", "RHOB"]
    for case, suffix in CASE_SUFFIX.items():
        names += [f"VP_{suffix}", f"VS_{suffix}", f"RHOB_{suffix}"]
    return names + ["GR", "VSH", "PHI", "SW"]


def build_logs(step=STEP, layers=LAYERS, noise=True, seed=NOISE_SEED):
    """Return a dict of canonical logs for the layer-cake model."""
    top = layers[0][0]
    base = layers[-1][1]
    depth = np.arange(top, base + step / 2, step)

    names = _curve_names()
    curves = {k: np.full(depth.size, np.nan) for k in names}

    for z_top, z_base, layer in layers:
        sel = (depth >= z_top) & (depth < z_base)
        vp, vs, rho = layer.cases["in situ"]
        values = {"VP": vp, "VS": vs, "RHOB": rho}
        for case, suffix in CASE_SUFFIX.items():
            c_vp, c_vs, c_rho = layer.cases[case]
            values[f"VP_{suffix}"] = c_vp
            values[f"VS_{suffix}"] = c_vs
            values[f"RHOB_{suffix}"] = c_rho
        values.update({"GR": layer.gr, "VSH": layer.vsh,
                       "PHI": layer.phi, "SW": layer.sw})
        for key, value in values.items():
            curves[key][sel] = value

    for key in names:  # the final sample sits exactly on `base`
        column = curves[key]
        curves[key] = np.where(np.isnan(column), column[~np.isnan(column)][-1], column)

    if noise:
        rng = np.random.default_rng(seed)
        # One noise realisation per measured curve, shared by all of that
        # curve's fluid cases: the substituted logs are derived from the same
        # measurement, so they carry the same noise rather than independent
        # draws.  Independent draws would leave a non-reservoir shale looking
        # different from case to case, and would shift the integrated
        # time axis between cases.
        draws = {base: rng.normal(0.0, sigma, depth.size) for base, sigma in NOISE.items()}
        for key in names:
            base = key.split("_")[0]
            if base in draws:
                curves[key] = curves[key] + draws[base]

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

    def unit_of(name):
        if name.startswith(("VP", "VS")):
            return "M/S"
        if name.startswith("RHOB"):
            return "G/C3"
        return {"GR": "GAPI"}.get(name, "V/V")

    case_of = {suffix: case for case, suffix in CASE_SUFFIX.items()}

    def descr_of(name):
        parts = name.split("_")
        base = {
            "VP": "Compressional velocity", "VS": "Shear velocity",
            "RHOB": "Bulk density", "GR": "Gamma ray", "VSH": "Shale volume",
            "PHI": "Effective porosity", "SW": "Water saturation",
        }[parts[0]]
        if len(parts) > 1:
            return f"{base} - {case_of[parts[1]]} substituted"
        return base

    las.append_curve("DEPT", curves["DEPTH"], unit="M", descr="Measured depth")
    for key in _curve_names():
        las.append_curve(key, curves[key], unit=unit_of(key), descr=descr_of(key))

    las.header["Other"] = (
        "Synthetic AVO demonstration well. Shale / Class III gas sand / shale, with a "
        "brine sand and a cemented streak below. Carries brine, oil and gas fluid "
        "cases (_BR, _OIL, _GAS) substituted offline with Gassmann; the AVO QI "
        "toolkit itself performs no fluid substitution. Not real data."
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
