# Rock Physics Toolkit — AVO & QI Well Analysis

A Streamlit application that takes a well **already containing Vp, Vs and
RHOB** and carries it through the standard quantitative-interpretation
workflow: rock-physics crossplots, a synthetic angle gather, and an
intercept–gradient AVO classification of every reflector.

A well that arrives with its fluid cases already substituted is the normal
path and is read as-is. The Rock Physics page can also **model** them here: a
forward model driven from `VSH`, `PHIT` and `SW`, and Gassmann fluid
substitution at Batzle-Wang reservoir conditions. Anything computed is
labelled *(computed)* wherever it appears.

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
| **Load & QC** | LAS / CSV / Excel loading, curve assignment with the units the header declares (or a magnitude sniff where it is silent), the **depth reference** — TVD from a deviation survey by minimum curvature, then TVDSS and TVDBML — a question about whether the file's own **petrophysical interpretation** should be kept or recomputed here, zonation from a zone curve or a tops list, and QC: null sentinels, coverage, plausible-range checks, spike detection and repair, depth-axis checks, and the elastic consistency tests — Vs faster than Vp, Vp/Vs below √2, Poisson outside its bounds. Ends in a depth window that the rest of the toolkit then works on. |
| **Data & Crossplots** | Upload, mnemonic remap and fluid-case selection, log tracks, and the QI crossplots: AI vs Vp/Vs, λρ–μρ (LMR), IP–IS, Poisson vs AI, and EEI with a χ sweep that reports the χ best correlated with Sw, Vsh or φ. |
| **Synthetic Gather** | Ricker / Ormsby / uploaded wavelet, exact Zoeppritz or Aki-Richards reflectivity, variable-density or wiggle gather display, near / mid / far and full stacks, and CSV / NPY / SEG-Y export. |
| **AVO Classification** | Reflectors picked from the full stack itself — every turning point above the amplitude cut is an event — then per-reflector A and B fitted by both Shuey and Aki-Richards, class I / IIp / IIn / III / IV assignment, an optional Monte Carlo that turns each label into a probability, the A–B crossplot with shaded class regions and a robust background trend, a reflector table, a filter on the interface *pair* so only shale-over-sand tops need be kept, and a clickable trace whose extrema are coloured by class and drive the per-reflector detail. That detail panel puts VSH, Vp, Vs and RHOB and the angle gather on the trace's own two-way-time axis, draws the trace variable-area with troughs red and peaks blue, and shades the two half-lobes the selected reflector's layers were averaged over. Layer properties come from each reflector's own lobe on the full stack — the upper half of the trough or peak gives the layer above, the lower half the layer below — a tuned-versus-untuned section shows what bed thickness does to each reflector's class, and a **wedge model** seeded from the selected event thins that reservoir from thick to nothing to give the tuning curve and the apparent-versus-true thickness. Ends with a one-click **self-contained HTML report**. |
| **Rock Physics** | A **rock physics template** — constant-porosity and constant-Sw curves on the AI vs Vp/Vs crossplot, so the axes read as porosity and saturation rather than merely "softer", built by running the per-sample forward model over the grid so the template and the prediction cannot drift apart. Diagnostic model overlays: Castagna mudrock and Greenberg-Castagna Vp–Vs trends, Gardner with a fitted exponent, velocity–porosity against Wyllie / Raymer-Hunt-Gardner and the Hashin-Shtrikman bounds, and K/μ vs porosity against the saturated bounds plus Hertz-Mindlin soft-sand, stiff-sand and critical-porosity frames — raised dry-to-saturated through Gassmann on request. Then a **forward model** driven per-sample from VSH, PHIT and SW with a misfit readout, an optional **Monte Carlo** P10–P90 band, and **fluid substitution** at Batzle-Wang reservoir conditions. |

## Layout

```
avo_qi/
├── app.py                      # Streamlit entry + landing
├── ui.py                       # shared sidebar, session state, Plotly helpers
├── report.py                   # the self-contained HTML report
├── SPEC.md                     # the build specification
├── io/loader.py                # LAS + CSV/Excel, mnemonic map, unit standardise
├── core/
│   ├── reflectivity.py         # Zoeppritz + Aki-Richards
│   ├── wavelet.py              # Ricker, Ormsby, user wavelet ingest, spectrum
│   ├── synthetic.py            # RC series + gather assembly
│   ├── avo.py                  # A/B fits + classifier + background trend
│   ├── attributes.py           # AI, SI, Vp/Vs, Poisson, LMR, EEI
│   ├── rockphysics.py          # bounds, trends, dry-frame granular models
│   ├── blocking.py             # lobe windows, Backus upscaling to seismic resolution
│   ├── lithology.py            # VSH cutoffs, lithology pairs, GR transforms
│   ├── mixing.py               # fluid and mineral mixing laws
│   ├── gassmann.py             # fluid substitution, with a per-sample validity mask
│   ├── fluids.py               # Batzle-Wang K and rho at reservoir P and T
│   ├── petro.py                # per-sample forward model, rock physics template
│   ├── misfit.py               # bounds checks and predicted-vs-measured residuals
│   ├── uncertainty.py          # Monte Carlo priors, bands and AVO class odds
│   ├── zones.py                # zonation from a LAS curve or a tops list
│   ├── petrophysics.py         # density/neutron porosity, Archie, Simandoux
│   ├── depth.py                # minimum curvature, TVD, TVDSS, TVDBML
│   ├── qc.py                   # nulls, ranges, spikes, elastic consistency
│   └── tuning.py               # tuned vs untuned AVO, wedge model, apparent thickness
├── pages/                      # the five Streamlit pages
├── sample_data/                # demo_well.las and its generator
└── tests/                      # acceptance + smoke tests
    └── data/15_9_19_A.las   # a real well, for the tests that need mess
```

`core/` is dependency-light — numpy and scipy only, with pandas used just to
assemble the reflector table. It imports no Streamlit and no plotting library,
so it stays unit-testable on its own.

## Mixing laws

**Fluid mixing** asks what a pore holding brine, oil and gas behaves like, and
the answer depends on how the phases are arranged. Finely mixed they share a
pressure and the moduli average harmonically (**Wood**) — a few percent gas
drops a brine-filled modulus by an order of magnitude. Segregated into patches
they stiffen independently and average arithmetically (**patchy**) — the same
gas barely moves it. **Brie** parks a saturation between the two, and **Hill**
takes the blunt midpoint. Density is always the volume-weighted average: mass
adds however the phases are arranged, so only the moduli need a law.

**Mineral mixing** takes any number of minerals with **Voigt**, **Reuss**,
their **Hill** average, or the narrower **Hashin-Shtrikman-Walpole** bounds
and their average. The n-phase Walpole form reduces exactly to the two-phase
Hashin-Shtrikman bounds, which is asserted in the tests.

Both are on the *Rock Physics* page under **Mineral matrix** and **Pore
fluid**, with the same saturations shown under every law so the spread is
visible rather than hidden behind one number.

## Blocking to the lobe

An interface coefficient taken from two adjacent log samples is the true layer
contrast only when the boundary is a step. Real boundaries are gradational, and
then the contrast splits across several samples: no single interface carries it
and the untuned response comes out far too weak. On the demo well the gap
reaches **0.25 in Rpp**.

The AVO Classification page therefore always blocks, and blocks on the
reflector's **own lobe** on the full stack. A reflector shows up as a trough or
a peak running from one zero crossing to the next; its **upper half**, from the
crossing above down to the extremum, is what the layer above produced, and its
**lower half** belongs to the layer beneath. The logs are averaged over each —
by Backus, which is the correct elastic upscaling, or by an arithmetic mean.

The window is measured on the data rather than assumed from the wavelet, so it
narrows where interference squeezes the lobe and opens where the reflector
stands alone. On the demo well the median lobe spans 17 samples against the 26
of the fixed half cycle it replaces, which is why contrasts come out sharper.

Two consequences are surfaced rather than hidden. A lobe only one sample wide
cannot be halved, and one with no zero crossing in range cannot be bounded;
either falls back to the fixed half-cycle window and is marked `fixed window`
in the reflector table. And two turning points sharing one lobe — a shoulder
rather than a lobe of its own — get the same blocked layers and the same A and
B, because they are not separable at that bandwidth; reporting different
answers for them would be inventing resolution the data does not have.

## The trace picks the reflectors

A well log knows about every interface; a seismic trace shows only what its
bandwidth resolves. Picking reflectors off the logs and then hunting for the
amplitude each one produced gets that backwards, and it shows: on a real North
Sea well (15/9-19-A) a |R| threshold found **269 interfaces**, of which **114 —
42% — produced no turning point of their own on the full stack**. Their AVO
class was real interface physics but not a pickable answer: the amplitude at
that time belonged to a neighbour.

So the **trace decides where the reflectors are**. Every turning point on the
full stack above the amplitude cut is an event, and the logs are asked only
what the rock is doing there. The same well now gives **25 events, every one of
them on a lobe of its own**. Two consequences follow, and both are the point:

* **A reflector buried in a neighbour's lobe is impossible** — the neighbour
  *is* the event.
* **A thin bed gives one event, not two.** Where a top and a base interfere
  into a single trough, that trough is what the seismic shows and what you
  could pick; splitting it into two answers would be inventing resolution.

The cut is a fraction of the strongest event on the trace rather than an
absolute level, because trace amplitude scales with the wavelet and carries no
fixed units — the same absolute number would mean quite different things at two
peak frequencies.

The lithology pair is read the same way. `lobe_lithology` takes the commonest
label over each half-lobe rather than the two samples nearest the extremum, so
"shale over sand" names the rock the intercept and gradient actually came from;
a single sample of silt at the boundary no longer renames a layer the wave saw
as shale.

## The report

A CSV moves numbers well and a *result* badly: it carries none of the settings
that produced it, none of the figures, and none of the caveats. Six months on,
nothing in one says which wavelet was used, what the amplitude cut was, or
which events a filter removed.

The **Build report** button at the foot of the AVO Classification page writes a
single HTML file that answers those first — provenance and settings, then the
wavelet and its spectrum, the event and class counts, the A–B crossplot, the
full reflector table and the tuning summary.

It is genuinely self-contained: Plotly is inlined once rather than pulled from
a CDN, so the file opens with no network and still works when a CDN version
moves on. That costs about 5 MB, which is the right trade for something meant
to be emailed and archived, and `no_external_references` exists so a test holds
the property rather than trusting it. Verified by opening the report in a
browser with networking disabled: both figures render and zero requests fail.

That check has one subtlety worth knowing, because it read as a bug first. The
inlined Plotly bundle carries map support, and that support has OpenStreetMap
and MapLibre attribution links and an unpkg icon URL written into its source. A
scan that does not exclude script *bodies* finds those strings and calls a
perfectly self-contained file external. Nothing in a report draws a map, so
none of it is ever fetched — but a `<script src=...>` pointing elsewhere would
genuinely break offline, so that is still checked on its own and never excused.

## Does the file already carry an interpretation?

VSH, PHI and SW drive the lithology classes, the zone summary's net-to-gross,
the forward model and every fluid substitution, so where they come from decides
a great deal — and the toolkit asks rather than deciding. *4 · Petrophysics* on
the Load & QC page reports which of the three the file carries and offers three
answers: keep them, compute only what is missing, or compute all three here.

**Keeping them is usually right.** An interpretation that arrived with the well
was made with core, pressures and local calibration, none of which is in the
toolkit. Computing is for the well that carries only raw logs, where the
alternative is nothing at all.

The transforms are the standard first pass — `vsh_from_gr`, density or
density-neutron porosity, Archie or Simandoux — with every parameter exposed
rather than buried. The defaults are read off **the well's own parameter
curves** where it has them (`RHOMA`, `RHOFL`, `RW`, `M`, `N`, `GRMIN`,
`GRMAX`), because those are what its interpretation was actually made with. On
15/9-19-A that is the difference between a textbook 2.65 / 1.00 g/cc and the
well's own 2.66 / 0.80, and recomputing with the latter reproduces the file's
own porosity to a median difference of **0.0002** (correlation 0.94); Sw comes
back at 0.97 and VSH at 0.96.

Three things keep it honest:

- **The file's curves are never destroyed.** They are snapshotted before
  anything is computed into the same columns, so "use what the file carries" is
  a real revert and the file-versus-computed comparison stays available.
- **What is computed says so**, in the status table and in the sidebar next to
  the well's name. A density porosity built on the wrong matrix density is
  wrong everywhere downstream, in the same direction, and silently.
- **What cannot be computed is explained.** The demo well carries no
  resistivity, so no saturation is derived from it and the page says why rather
  than quietly leaving the field empty.

## Depth references

Measured depth runs along the hole from a rig floor, so it is neither
comparable between wells nor the depth anything in the earth responds to. The
toolkit carries four references and each event in the reflector table gets all
of them:

| Reference | Measured from | What it is for |
|---|---|---|
| MD | drilling datum, along hole | where a sample sits in *this* well |
| TVD | drilling datum, vertically | removing the hole's deviation |
| TVDSS | mean sea level | comparing two wells at all |
| TVDBML | seabed | compaction — and so porosity and velocity — trends |

All four increase downwards: TVDSS is positive below sea level, not a negative
elevation. Software that plots subsea depth on a negative axis is using the
opposite sign.

*3 · Depth reference* on the Load & QC page resolves them, in this order:

1. **Curves in the file win.** A TVD or TVDSS the file already carries was made
   with the survey and the datum the well was actually drilled on, and nothing
   reconstructed here beats that.
2. **A deviation survey** — a LAS, CSV or Excel file of measured depth,
   inclination and azimuth; a survey is a depth-indexed set of curves, so a LAS
   is as natural a format for it as a spreadsheet — gives TVD by **minimum
   curvature** — a circular arc between
   stations rather than a straight line. Log samples are placed inside a survey
   interval by interpolating the hole's attitude and taking one curvature step
   from the station above, not by interpolating TVD linearly between stations.
3. **A declared vertical well** takes TVD = MD. This is offered as an explicit
   choice rather than a default, because a 30° hole at 4000 m MD is about
   200 m shallower than its measured depth.

TVDSS then needs the height of the drilling datum above mean sea level, and
TVDBML additionally the water depth. Both are read from the LAS header where it
carries them (`EKB`, `KB`, `WD` and their usual aliases) and asked for where it
does not — which is most of the time: 15/9-19-A has `EKB`, `EGL`, `KB` and `GL`
entries and every one of them is empty.

What cannot be resolved is left out. A reference with no datum behind it is
**removed rather than written as a column of nulls**, so "no TVDSS here" reads
as an absent curve everywhere downstream instead of a curve that is somehow all
blank. A plausible-looking subsea depth built on an assumed rig-floor height is
worse than none.

## Zonation

Wells arrive zoned two ways, and both work here. A LAS carrying a discrete
`ZONE`, `FORMATION`, `MARKER` or `UNIT` curve is recognised on load and turned
into named intervals. Most wells carry no such curve — 15/9-19-A does not —
which used to leave the zone filter, the zone-boundary flag and every per-zone
summary with nothing to work from; **formation tops** entered or uploaded on
the *Load & QC* page now fill that gap. An uploaded zonation may be a **LAS**
as well as a CSV or Excel tops list: a LAS data section is numeric and has
nowhere to put a name, so it carries the zonation as a discrete curve, and the
codes stand in as their own names until they are typed over.

Which shape a table is, is decided on its **content, not its column names**. A
tops list never repeats a zone on consecutive rows — that would be a zone
interrupted by nothing — so a label that repeats means a curve. Deciding on
names instead looks fine until a zone curve arrives with its depth mnemonic
spelled `DEPTH` or `MD`, which a tops reader accepts and turns into several
thousand one-sample tops. Each zone runs to the next top and the
deepest to the bottom of the well. Tops are cleared when another well is
loaded, because a top is a depth in the well it was typed against.

An event is flagged as sitting on a zone boundary when a top falls anywhere
inside **its lobe**, not merely between the two samples at its extremum. With
reflectors picked off the trace there are few of them and a top almost never
lands in that one-sample gap, so the older test went quiet and stopped meaning
anything. The lobe is the event's zone of influence, which is the right
question to ask of a reflector.

Codes map to names through the sidebar; an unmapped code stands in as its own
name, so a zonation with no legend is still usable. A tops list — names and depths, from a dict or a
DataFrame — works too.

Zone codes are labels, not measurements, so intervals are built from where the
code changes rather than by interpolating, and a zone that reappears deeper
stays a separate interval instead of merging with its earlier occurrence.

The sidebar filters every page by zone, and each reflector carries the zone
above it, the zone below, and whether it **is** a zone boundary — usually the
reflector that matters most.

### Zone summary

A zonation that only filters is half a zonation. The **Zone summary** on the
AVO Classification page turns it into an answer: per zone, gross thickness,
net and pay against VSH / porosity / Sw cutoffs, thickness-weighted log
averages over both the zone and its net interval, and — from the events picked
on the trace — how many fall in the zone, their class mix, and the one sitting
furthest below the background trend.

Three things about it are deliberate.

**It is measured in depth, not in time.** A fast layer occupies fewer time
samples per metre, so a net-to-gross counted on the time trace would be biased
by velocity. Thicknesses and averages come off the depth log; only the event
side comes from the trace.

**Missing curves lower coverage rather than net.** A sample with no Sw cannot
pass an Sw cutoff, but it is not demonstrably non-pay either. `net_coverage`
and `pay_coverage` say how much of the zone could be judged at all, so a zone
with no Sw curve reads as *unknown* pay instead of as zero pay — which matters
on a well like 15/9-19-A, logged for Sw over half its length.

**An event is counted on either side.** A reservoir top has its upper lobe in
the seal, so counting only the upper side would file the best event a
reservoir has under the shale above it. A boundary event belongs to the pair
and appears in both.

A zone that recurs down the well sums the rock it occupies rather than
spanning from its first sample to its last — the same reasoning that keeps two
occurrences of a zone as separate intervals.

## Fluid cases

A well that arrives with substitution already done — `VP_BR`, `VS_OIL`,
`RHOB_GAS` and friends — is recognised automatically and every case becomes
selectable in the sidebar. Cases substituted *here* are written in the same
scheme and join the same selector, marked **(computed)**. Recognised suffixes
cover brine (`_BR`, `_BRINE`,
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

## The forward model, and where the numbers come from

The bounds and frame curves are set by hand and stay **independent of the
elastic logs**. That is the point of them: a model whose inputs are derived
from the measurements it is tested against cannot disagree with them, and a
model that cannot disagree is not a diagnostic. Where the well misses an
overlay, the page counts the miss rather than leaving it to the eye — *"93% of
1,051 samples lie inside the bounds; the worst breach is 1.8 GPa below the
lower bound at 2104 m."*

The **Forward model** tab is the one place the well drives the model, and it
is not circular. It runs

```
VSH  → mineral mix   → K_ma, G_ma, ρ_ma
SW   → fluid mix     → K_fl, ρ_fl
PHIT → dry frame → Gassmann → K_sat ;  mass balance → ρ  →  Vp, Vs
```

and compares the predicted logs against the measured ones. `VSH` comes from
gamma ray and `SW` from resistivity — neither has seen a velocity — so a
predicted Vp that misses the log is a real disagreement.

`PHIT` is the exception, and the page checks for it. Density porosity is
computed *from* RHOB, so predicting RHOB from it always succeeds and means
nothing. `porosity_provenance` fits a matrix and a fluid density to the
(PHIT, RHOB) pair and inspects the residual; fitting the constants rather than
assuming them catches density porosity whatever values the petrophysicist
used. When it fires, the density comparison is marked circular and set aside,
and the Vp and Vs comparisons — which stay valid — carry the result.

### Uncertainty

Every number above is a point answer standing on inputs that are not points.
`core/uncertainty.py` samples them and runs the model many times, so the
answers come back as distributions instead.

Two kinds of uncertainty are kept apart, because conflating them is wrong.
**Log noise** is per sample — each depth's VSH carries its own measurement
error. **Model parameters** are per realisation: a critical porosity is a
property of the rock type, and drawing a fresh one at every depth would model
a well whose grain packing changes every 15 cm, which is not the uncertainty
anyone means.

Correlations between logs default to **zero**. VSH and porosity really are
anti-correlated in most clastics, and saying so narrows the cloud
considerably — but that is a claim about the rock, so it is a control rather
than a default.

On **Rock Physics → Forward model** this draws a P10–P90 band around the
predicted logs. A band that reaches further one way than the other is not a
drawing artefact: a log sitting against a physical limit — a shale at SW 1.0 —
can only be perturbed away from it, so its uncertainty is genuinely one-sided.

The payoff is on **AVO Classification**. A class label reads as a fact when it
is really the answer to *where do the intercept and gradient land*, and both
come from logs with a measurement error. Perturbing the logs within that error
and reclassifying each time turns the label into odds:

> Median confidence 100% · Least confident 34% · Ambiguous 2 of 9

The median is deliberately shown next to the minimum, because on its own it
hides the reflectors that matter. Where the most likely class holds less than
half the realisations, no class is really being asserted — and where the modal
class differs from the deterministic label, the reflector is sitting on a
boundary, which is the thing worth knowing. The probabilities travel into the
reflector table and its CSV, so the odds stay attached to the label.

Two implementation notes. The two-term Shuey fit has a closed form, so
`batched_shuey_fit` solves tens of thousands of reflectors with array
reductions rather than a per-row `lstsq` — the same least-squares answer, not
an approximation, and the tests pin it against `shuey_fit` to keep it that
way. It is the difference between 8 seconds and 0.05. `batched_aki_richards`
is pinned against the scalar `aki_richards_rpp` the same way.

### Fluid substitution

`core/gassmann.py` runs Gassmann in both directions, and
`core/fluids.py` supplies Batzle-Wang properties at reservoir pressure and
temperature — the fixed table is a room-condition approximation, and at 30 MPa
and 90 °C a gas is roughly five times stiffer and six times denser than its
entry in it.

Substituting writes ordinary fluid cases (`VP_GAS` and friends), so every other
page picks them up through machinery that already existed. What is *not* left
implicit is where they came from: computed cases are recorded in
`WellData.computed_cases`, shown as **(computed)** in every case selector, and
the page warns before replacing a case that was loaded from the file.

**Failure is reported, never hidden.** Running Gassmann backwards from a
measured log routinely returns a negative dry frame, which does not mean "no
answer" — it means the porosity, the mineral and the velocities disagree,
because no real rock is softer than a vacuum. Every routine returns the values
*and* a per-sample reason, and the demo well's original brine sand
(2500 / 1450 / 2.30 at φ 0.26, needing K_dry = −1.6 GPa) is pinned as a
regression test for exactly that.

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
| Depth | m | ft (MD, TVD, TVDSS and TVDBML alike) |
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

`test_real_well.py` runs the whole thing against a **real** well —
`tests/data/15_9_19_A.las`, a North Sea well of 3905 samples over 3500–4095 m
MD, with the gaps real logs have: VSH on 2812 samples of 3905, SW on 1965,
RHOB carrying null sentinels. The bundled demo well is a clean three-layer
model and is the right fixture for pinning a known answer; it is the wrong one
for finding out what breaks. Every defect in the blocking and
reflector-picking work came from this well and none from the demo — a
reflector split on a neighbour's lobe, a NaN polarity cast to `INT_MIN`, three
reflectors reaching the table and the CSV with **non-finite A and B**, and 42%
of the reflector table describing interfaces the seismic could not separate.
Most of these tests assert invariants; the two that pin numbers say so, and
exist to make a change in the defaults visible rather than silent.
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
- Gassmann (1951) — the fluid-substitution relation. Batzle & Wang (1992),
  *Geophysics* 57, 1396–1408 — brine, oil and gas properties at reservoir
  pressure and temperature.
- Wood (1955) — the harmonic fluid average. Brie et al. (1995) — the empirical
  curve between the uniform and patchy limits. Walpole (1966) — the n-phase
  extension of the Hashin-Shtrikman bounds.
