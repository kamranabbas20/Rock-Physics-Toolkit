# Rock Physics Toolkit — AVO & QI Well Analysis

A Streamlit application that takes a well **already containing Vp, Vs and
RHOB** and carries it through the standard quantitative-interpretation
workflow: rock-physics crossplots, a synthetic angle gather, and an
intercept–gradient AVO classification of every reflector.

Fluid substitution is deliberately **out of scope** — no Gassmann, no
Batzle-Wang. The input well is taken as-is; if substitution is needed it
belongs in a separate upstream tool.

The full build brief is [`avo_qi/SPEC.md`](avo_qi/SPEC.md).

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run avo_qi/app.py
```

Click **Load demo well** in the sidebar to work with the bundled three-layer
model, or upload a LAS, CSV or Excel well on the *Data & Crossplots* page.

## Pages

| Page | What it does |
|------|--------------|
| **Data & Crossplots** | Upload and mnemonic remap, log tracks, and the QI crossplots: AI vs Vp/Vs, λρ–μρ (LMR), IP–IS, Poisson vs AI, and EEI with a χ sweep that reports the χ best correlated with Sw, Vsh or φ. |
| **Synthetic Gather** | Ricker / Ormsby / uploaded wavelet, exact Zoeppritz or Aki-Richards reflectivity, variable-density or wiggle gather display, near / mid / far and full stacks, and CSV / NPY / SEG-Y export. |
| **AVO Classification** | Per-reflector A and B fitted by both Shuey and Aki-Richards, class I / IIp / IIn / III / IV assignment, the A–B crossplot with shaded class regions and a robust background trend, a reflector table, and per-reflector amplitude-vs-angle curves. |
| **Rock Physics** | Diagnostic model overlays: Castagna mudrock and Greenberg-Castagna Vp–Vs trends, Gardner with a fitted exponent, velocity–porosity against Wyllie / Raymer-Hunt-Gardner and the Hashin-Shtrikman bounds, and K/μ vs porosity against the saturated bounds plus dry-frame Hertz-Mindlin soft-sand, stiff-sand and critical-porosity models. |

## Layout

```
avo_qi/
├── app.py                      # Streamlit entry + landing
├── ui.py                       # shared sidebar, session state, Plotly helpers
├── SPEC.md                     # the build specification
├── io/loader.py                # LAS + CSV/Excel, mnemonic map, unit standardise
├── core/
│   ├── reflectivity.py         # Zoeppritz + Aki-Richards
│   ├── wavelet.py              # Ricker, Ormsby, user wavelet ingest
│   ├── synthetic.py            # RC series + gather assembly
│   ├── avo.py                  # A/B fits + classifier + background trend
│   ├── attributes.py           # AI, SI, Vp/Vs, Poisson, LMR, EEI
│   └── rockphysics.py          # bounds, trends, dry-frame granular models
├── pages/                      # the four Streamlit pages
├── sample_data/                # demo_well.las and its generator
└── tests/                      # acceptance + smoke tests
```

`core/` is dependency-light — numpy and scipy only, with pandas used just to
assemble the reflector table. It imports no Streamlit and no plotting library,
so it stays unit-testable on its own.

## No fluid substitution

There is no Gassmann and no Batzle-Wang anywhere in this toolkit, by design —
the input well is taken as-is. The Rock Physics page stays inside that
boundary: the Hashin-Shtrikman and Voigt-Reuss-Hill bounds are computed on a
mineral-plus-fluid mixture, so they bracket the **saturated** rock directly
without a substitution step, and the granular models (Hertz-Mindlin, soft
sand, stiff sand, critical porosity) are **dry-frame** curves, labelled as
such wherever they are drawn. Raising a dry frame to a saturated one needs
Gassmann, so the page says so rather than doing it.

A consequence worth knowing when reading the demo well: its elastic logs are
fixed by the validated AVO cross-check in SPEC.md §4.1 and describe a shallow,
poorly consolidated section. They are softer than a consolidated quartz/brine
model predicts at their porosities, so several demo layers plot below the
suspension bound. That is the diagnostic doing its job — switching the pore
fluid to gas brings the gas sand back inside its bounds.

## Units

All conversion happens in `io/loader.py` and the Streamlit widgets; `core/`
never converts.

| Quantity | Core unit | Converted on load |
|----------|-----------|-------------------|
| Velocity | m/s | ft/s, km/s, and sonic slowness (µs/ft, µs/m) |
| Density | g/cc | kg/m³ |
| Depth | m | ft |
| Time | TWT s | integrated from Vp when only MD is given |
| Angle | degrees | — |

## Tests

```bash
pytest
```

`avo_qi/tests/test_core.py` holds the seven acceptance tests from SPEC.md §6 —
Zoeppritz vs Aki-Richards agreement, normal-incidence equivalence to the AI
reflectivity, Shuey A/B recovery, the classifier truth table, the three-layer
gather signature, wavelet zero-phase behaviour, and both-fit consistency.
`test_rockphysics.py` covers the diagnostic models: bound ordering
(Reuss ≤ HS⁻ ≤ HS⁺ ≤ Voigt), end-member collapse, the frame models bracketing
each other and meeting at the Hertz-Mindlin pack, and the empirical trends.
`test_app_smoke.py` runs each Streamlit page headless against the demo well
and is skipped if Streamlit is not installed.

## References

- Aki & Richards (2002), *Quantitative Seismology*, 2nd ed. — reflectivity.
- Rutherford & Williams (1989); Castagna & Swan (1997) — AVO classes, Class IV
  and the A–B background trend.
- Shuey (1985) — the two-term `R(θ) = A + B sin²θ` intercept–gradient form.
- Whitcombe (2002) — extended elastic impedance.
- Hashin & Shtrikman (1963); Mavko, Mukerji & Dvorkin, *The Rock Physics
  Handbook* — mixture bounds and the granular frame models.
- Castagna, Batzle & Eastwood (1985); Greenberg & Castagna (1992) — Vp–Vs
  trends. Nur et al. (1998) — critical porosity. Dvorkin & Nur (1996) — the
  soft- and stiff-sand models.
