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
model — which carries brine, oil and gas cases alongside its in-situ logs —
or upload a LAS, CSV or Excel well on the *Data & Crossplots* page.

## Pages

| Page | What it does |
|------|--------------|
| **Load & QC** | LAS / CSV / Excel loading, curve assignment with the units the header declares (or a magnitude sniff where it is silent), and QC: null sentinels, coverage, plausible-range checks, spike detection and repair, depth-axis checks, and the elastic consistency tests — Vs faster than Vp, Vp/Vs below √2, Poisson outside its bounds. Ends in a depth window that the rest of the toolkit then works on. |
| **Data & Crossplots** | Upload, mnemonic remap and fluid-case selection, log tracks, and the QI crossplots: AI vs Vp/Vs, λρ–μρ (LMR), IP–IS, Poisson vs AI, and EEI with a χ sweep that reports the χ best correlated with Sw, Vsh or φ. |
| **Synthetic Gather** | Ricker / Ormsby / uploaded wavelet, exact Zoeppritz or Aki-Richards reflectivity, variable-density or wiggle gather display, near / mid / far and full stacks, and CSV / NPY / SEG-Y export. |
| **AVO Classification** | Per-reflector A and B fitted by both Shuey and Aki-Richards, class I / IIp / IIn / III / IV assignment, the A–B crossplot with shaded class regions and a robust background trend, a reflector table, and a clickable trace whose extrema are coloured by class and drive the per-reflector detail. Layer properties come from half-cycle blocked layers by default rather than two adjacent samples, and a tuned-versus-untuned section shows what bed thickness does to each reflector's class. |
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
│   ├── rockphysics.py          # bounds, trends, dry-frame granular models
│   ├── blocking.py             # half-cycle upscaling to seismic resolution
│   ├── lithology.py            # VSH cutoffs, lithology pairs, GR transforms
│   ├── qc.py                   # nulls, ranges, spikes, elastic consistency
│   └── tuning.py               # tuned vs untuned AVO, wedge model
├── pages/                      # the five Streamlit pages
├── sample_data/                # demo_well.las and its generator
└── tests/                      # acceptance + smoke tests
```

`core/` is dependency-light — numpy and scipy only, with pandas used just to
assemble the reflector table. It imports no Streamlit and no plotting library,
so it stays unit-testable on its own.

## Fluid cases

A well that arrives with substitution already done — `VP_BR`, `VS_OIL`,
`RHOB_GAS` and friends — is recognised automatically and every case becomes
selectable in the sidebar. Recognised suffixes cover brine (`_BR`, `_BRINE`,
`_WET`, ...), oil, gas and in situ, with or without the underscore; a case is
only kept when all three of Vp, Vs and RHOB are present for it. Curves like
`VSH` are never mistaken for a shear log, because a suffix has to be a known
fluid token.

What the cases unlock:

- **Data & Crossplots** — overlay every case on the QI crossplots to see the
  fluid vector in AI–Vp/Vs space.
- **Synthetic Gather** — a gather per case, a difference gather between any
  two, and their full stacks superimposed.
- **AVO Classification** — every reflector fitted in every case, an A–B
  crossplot with an arrow along each reflector's fluid vector, a table of the
  class each reflector takes in each case, and a list of the reflectors whose
  class changes with fluid at all.
- **Rock Physics** — the pore fluid used for the bounds follows the case.

One thing matters more than it looks: the **two-way-time axis is integrated
once, from the well's in-situ case, and every other case is resampled onto
that same grid.** Letting each case integrate its own Vp gives each one a
different time axis, so sample *i* is a different interface in each — the
reflectors silently misalign, and only the shallowest one, above the first
reservoir, still lines up. `reflector_avo` takes an explicit `samples`
argument for the same reason, so all cases are fitted at identical
interfaces rather than at whatever each happens to detect.

## Two things the classification gets right that are easy to get wrong

**Post-critical angles are excluded from the fit.** Where the lower layer is
fast enough, the critical angle falls inside the modelled angle range — a
cemented sand under shale goes critical near 34°. Past that the exact
Zoeppritz solution is complex, and the real continuation used to keep gathers
finite spikes upward. Fitting those points drags the gradient positive: on the
demo well's cemented streak it turns B = −0.55 into B = +0.31 and reports a
textbook Class I as background. `reflector_avo` masks angles at or beyond each
interface's critical angle before fitting (`mask_post_critical`, on by
default), reports the angle and how many angles survived, and the page shades
the excluded region.

**Reflector markers are matched by polarity, not by amplitude.** On the
clickable trace, each reflector's marker sits on the turning point *nearest*
the interface *whose sign matches* its normal-incidence coefficient. Taking
the largest amplitude in a window instead hands a weak reflector its loud
neighbour's lobe — three of the demo well's nine reflectors ended up on an
extremum of the opposite sign that way. A reflector with no turning point of
its own polarity in range is drawn hollow at the interface time, because its
amplitude is not an extremum and should not be read as one.

## No fluid substitution

There is no Gassmann and no Batzle-Wang anywhere in this toolkit, by design —
the input well is taken as-is, fluid cases included. The toolkit recognises
substituted curves; it never produces them. The Rock Physics page stays inside that
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

## Running fully offline

At runtime the toolkit needs no network at all. Every asset Streamlit serves
is local — no CDN, no Google Fonts, and plotly.js ships inside the Python
package. A test proves it rather than asserting it: it blocks every
non-loopback socket, then renders a full AVO page and checks nothing tried to
dial out.

The only step that needs a network is `pip install`. To remove that too, build
a wheelhouse once on a connected machine:

```bash
./scripts/make_offline_bundle.sh                  # for this machine
./scripts/make_offline_bundle.sh win_amd64 3.11   # or cross-build for Windows
```

Carry the repository across — wheelhouse included — and on the air-gapped
machine:

```bash
python3 -m venv .venv
.venv/bin/pip install --no-index --find-links wheelhouse -r requirements-lock.txt
.venv/bin/streamlit run avo_qi/app.py
```

`requirements-lock.txt` pins the full transitive set, so the install is
reproducible as well as offline. This path is tested end to end: install with
`--no-index`, run the suite, start the app.

## Yes, your logs go to the browser

Streamlit is client–server even when both halves are on your desk. Python does
the computing; the **browser does the drawing**, so the values behind every
plot and table are sent to it over a WebSocket on localhost.

Measured on a 3,938-sample well: **1.8 MB** travels to the browser, and 76% of
sampled Vp values are present verbatim as raw IEEE-754 float64 bytes in that
payload. This is not incidental — it is how the charts get drawn.

What that does and does not mean:

- It **never leaves the machine**. The socket is `ws://localhost`, and the
  browser capture below finds zero requests to any other host.
- The **browser process holds your logs in memory**. Browser devtools can read
  them, and so can **any extension with permission to read page content or
  network traffic**. That is the one real exposure in this design.
- Nothing is written to browser disk cache — WebSocket frames are not cached —
  but a browser crash dump could contain the values.

If the data is sensitive, run it in a browser profile with no extensions, or a
private window with extensions disabled, rather than the browser you use for
everything else. There is no way to draw an interactive chart without the
numbers reaching the renderer; the honest mitigation is controlling what else
is running in that renderer.

## Avoiding the browser entirely

If sending your logs to a browser is not acceptable, do not use the app. The
same analysis runs from a terminal and writes results to disk — no web server,
no WebSocket, no renderer holding your data:

```bash
python -m avo_qi.cli WELL.las --out results/
```

It writes the QC summary and per-sample flags, the standardised and
time-converted well, the reflector table with classes and lithology pairs, the
gather as CSV and NPY, the angle stacks, a text report, and a PDF of figures
drawn with Matplotlib's Agg backend — a file writer, not a display.

Useful options:

```bash
--case gas            # pick a fluid case
--top 2030 --base 2100  # analyse one interval
--despike             # repair spikes in Vp, Vs, RHOB
--drop-flagged        # discard samples that failed a QC check
--method aki_richards # or zoeppritz (default)
--freq 35             # Ricker peak frequency
--no-figures          # tables only, and no Matplotlib needed
```

`avo_qi/core/` imports neither Streamlit nor Plotly, which is what makes this
possible. A test in `avo_qi/tests/test_cli.py` runs the whole workflow with
every outbound connection blocked and asserts none is attempted, and a second
asserts that importing the CLI does not load a web stack at all.

## Verifying it yourself

Two checks, both runnable on your machine with your own well:

```bash
# Python side: blocks every non-loopback socket, then runs the analysis and
# renders a page. Fails if anything tries to dial out.
pytest avo_qi/tests/test_data_handling.py

# Browser side: drives a real browser through every page and records every
# request it makes. Reports any that leaves localhost.
pip install playwright && playwright install chromium
python scripts/verify_no_network.py path/to/your.las
```

The browser check on this repository captures 213 requests across the full
workflow — uploading a well and visiting every page — and none of them leave
localhost.

## Your data stays on your machine

Well data is usually proprietary, so the handling is deliberate and tested
(`avo_qi/tests/test_data_handling.py`):

- **Nothing is sent anywhere.** No module in `avo_qi/` imports a network
  client, and a test asserts it stays that way.
- **Uploads are parsed in memory.** A LAS you upload is never written to disk,
  so it cannot be left in the working directory and swept up by a later
  `git add`. `.gitignore` covers stray `.LAS` files and upload artefacts as a
  second line of defence, while keeping the demo well tracked.
- **Streamlit's telemetry is off.** Streamlit reports anonymous usage
  statistics to its makers by default. It does not send well data, but it is
  an outbound connection; the shipped `.streamlit/config.toml` disables it.
- **The server binds to localhost only.** Streamlit otherwise listens on every
  network interface, so anyone able to reach your machine could open the app
  and read the loaded well. Override deliberately if you mean to share it.

Loaded wells live in the Streamlit session in memory and disappear when the
process stops. Everything you export — CSV, NPY, SEG-Y — is a browser
download that you choose.

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
`test_qc.py` covers the QC checks, including a forged well carrying the
defects a real LAS arrives with — sonic in µs/ft, density in kg/m³, null
intervals, spikes and a bad shear section.
`test_fluid_cases.py` covers suffix detection (including the `VSH` trap), the
shared time axis, and the cross-case comparison.
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
