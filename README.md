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
| **Load & QC** | LAS / CSV / Excel loading, curve assignment with the units the header declares (or a magnitude sniff where it is silent), the **depth reference** — TVD from a deviation survey by minimum curvature, then TVDSS and TVDBML — a question about whether the file's own **petrophysical interpretation** should be kept or recomputed here, a **shear sonic** predicted where the well has none or only part of one and scored against whatever it does have, **fluid cases** assigned where the file carries them and modelled from industry-default fluids where it does not, zonation from a zone curve or a tops list, and QC: null sentinels, coverage, plausible-range checks, spike detection and repair, depth-axis checks, and the elastic consistency tests — Vs faster than Vp, Vp/Vs below √2, Poisson outside its bounds. Ends in a depth window that the rest of the toolkit then works on, and a **setup file** that saves every decision on the page — your interpretation, not your logs. |
| **Data & Crossplots** | Upload, mnemonic remap and fluid-case selection, log tracks, and the QI crossplots: AI vs Vp/Vs, λρ–μρ (LMR), IP–IS, Poisson vs AI, and EEI with a χ sweep that reports the χ best correlated with Sw, Vsh or φ. |
| **Synthetic Gather** | Ricker / Ormsby / uploaded wavelet, exact Zoeppritz or Aki-Richards reflectivity, variable-density or wiggle gather display, near / mid / far and full stacks, and CSV / NPY / SEG-Y export. |
| **AVO Classification** | Reflectors picked from the full stack itself — every turning point above the amplitude cut is an event — then per-reflector A and B fitted by both Shuey and Aki-Richards, class I / IIp / IIn / III / IV assignment, an optional Monte Carlo that turns each label into a probability, the A–B crossplot with shaded class regions and a robust background trend, a reflector table, a filter on the interface *pair* so only shale-over-sand tops need be kept, and a clickable trace whose extrema are coloured by class and drive the per-reflector detail. That detail panel puts VSH, Vp, Vs and RHOB and the angle gather on the trace's own two-way-time axis, draws the trace variable-area with troughs red and peaks blue, and shades the two half-lobes the selected reflector's layers were averaged over. Layer properties come from each reflector's own lobe on the full stack — the upper half of the trough or peak gives the layer above, the lower half the layer below — a tuned-versus-untuned section shows what bed thickness does to each reflector's class, and a **wedge model** seeded from the selected event thins that reservoir from thick to nothing to give the tuning curve and the apparent-versus-true thickness. Then **where the classes are** — small multiples on one shared, downward depth axis, the class on an axis of its own as well as in the colour, with a panel per property. Then **AVO attributes** — pseudo-shear reflectivity and the Smith-Gidlow fluid factor at each reflector's own background Vp/Vs, an anomaly ranking scored in the well's own scatter, and a χ sweep that finds the rotation of the A–B plane best correlated with a rock property. Then **class against property** — every reflector's φ, VSH, SW and net-to-gross averaged over the same two half-lobes its intercept and gradient were fitted from, every available property ranked by how well it separates the classes, and the one you pick drawn as a box and its own events. Ends with a one-click **self-contained HTML report** that carries every section above it, as configured on screen. |
| **Rock Physics** | A **rock physics template** — constant-porosity and constant-Sw curves on the AI vs Vp/Vs crossplot, so the axes read as porosity and saturation rather than merely "softer", built by running the per-sample forward model over the grid so the template and the prediction cannot drift apart. Diagnostic model overlays: Castagna mudrock and Greenberg-Castagna Vp–Vs trends, Gardner with a fitted exponent, velocity–porosity against Wyllie / Raymer-Hunt-Gardner and the Hashin-Shtrikman bounds, and K/μ vs porosity against the saturated bounds plus Hertz-Mindlin soft-sand, stiff-sand and critical-porosity frames — raised dry-to-saturated through Gassmann on request. Then a **forward model** driven per-sample from VSH, PHIT and SW with a misfit readout, an optional **Monte Carlo** P10–P90 band, and **fluid substitution** at Batzle-Wang reservoir conditions. |
| **Multi-well** | Every well in the library run through the *same* reflector pipeline and set side by side: an overview of what each well contributed, the intercept–gradient crossplot with all wells on it and a background trend fitted through the lot — a trend fitted in one hole is that hole's rock, fitted across several it is the field's — a curve of your choice against TVDSS or TVDBML, every picked event on one shared depth axis with colour for the class and shape for the well, the class mix per well, and events by zone using **each well's own** tops — then the anomaly ranking and the class-against-property panel over every well at once — each well measured against its own trend and scored in its own scatter, which is where a relationship stops being one hole's coincidence and a quiet well's best event stays visible beside a noisy one's. |

## Layout

```
avo_qi/
├── app.py                      # Streamlit entry + landing
├── ui.py                       # shared sidebar, session state, Plotly helpers
├── analysis.py                 # the reflector pipeline, shared by the pages that need it
├── project.py                  # the setup file: what you decided, without the well data
├── report.py                   # the self-contained HTML report
├── SPEC.md                     # the build specification
├── io/loader.py                # LAS + CSV/Excel, mnemonic map, unit standardise
├── core/
│   ├── reflectivity.py         # Zoeppritz + Aki-Richards
│   ├── wavelet.py              # Ricker, Ormsby, user wavelet ingest, spectrum
│   ├── synthetic.py            # RC series + gather assembly
│   ├── avo.py                  # A/B fits + classifier + background trend
│   ├── avo_attributes.py       # pseudo-Rs, fluid factor, chi rotation, anomaly ranking
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
│   ├── vs_prediction.py        # shear sonic where a well has none, and how well it went
│   ├── properties.py           # lobe-averaged rock properties, NTG, class dependence
│   ├── depth.py                # minimum curvature, TVD, TVDSS, TVDBML
│   ├── qc.py                   # nulls, ranges, spikes, elastic consistency
│   └── tuning.py               # tuned vs untuned AVO, wedge model, apparent thickness
├── pages/                      # the six Streamlit pages
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
single HTML file that answers those first — provenance and settings — and then
carries the whole page: the wavelet and its spectrum, the zonation, the event
and class counts, the A–B crossplot, *where the classes are*, the AVO
attributes with the anomaly ranking and the χ sweep, class against property,
the class probabilities where they were computed, the reflector table, the
tuning summary, the wedge model, and the fluid-case comparison. Thirteen
sections on a well that carries everything; fewer, and no empty headings, on
one that does not.

Two things make that work, and both are held by tests.

**The document is handed what the screen was handed.** The interpretive panels
return a `PanelReport` of the figures, tables and findings they drew, and the
report re-renders those rather than rebuilding them from defaults. You chose a
depth reference and four panels; the file shows those four. A report that
quietly disagrees with the page that produced it is worse than one that shows
less.

**The plotting library is inlined exactly once**, by a `Figures` allocator
rather than by whichever call site remembered to ask. That was fine to do by
hand at two figures. At ten — several of them conditional on what the well
carries, so *which* figure comes first depends on the well — it is a bug
waiting to happen: twice and the file doubles, never and it draws nothing at
all. Going from 2 figures to 10 cost 87 KB on a 4.9 MB file, because the
megabytes are the library and each further figure is only its own JSON.

It is genuinely self-contained: Plotly is inlined rather than pulled from a
CDN, so the file opens with no network and still works when a CDN version
moves on. That costs about 5 MB, which is the right trade for something meant
to be emailed and archived, and `no_external_references` exists so a test holds
the property rather than trusting it. Verified by opening the report in a real
browser: all ten figures render, and there are zero requests off the file and
zero console errors.

The report sits at the *foot* of the page rather than second-to-last, because
it reports on everything above it — including the fluid-case comparison, which
used to run after it and so could not be in it.

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
- **A curve is not reported where it was never logged.** Resampling the logs
  onto the time grid interpolates each curve **only within its own logged
  interval**. `np.interp` clamps outside its data range, and that is a quiet
  and expensive default: 15/9-19-A has no VSH, PHI or SW above 3666 m, a
  quarter of the interval, and the time frame came back with *zero* missing
  samples — a constant VSH of 0.599, a constant porosity of 0.200 and a
  constant SW of 0.559 filling rock nobody had interpreted. Nothing about that
  read as wrong downstream: the lithology pair said "silt over silt" with
  conviction and a net-to-gross could be measured over an interval with no
  petrophysics in it. Those five reflectors now report `undefined over
  undefined`, and the interface-pair filter leaves them out by default and
  says on screen that it has.

## Saving what you decided

Everything the Load & QC page establishes lives in the browser session, so
closing the tab loses it: the tops, the datum, the petrophysics choice, the
shear-sonic model, the fluid parameters, the cutoffs. On a real well that is
twenty minutes of re-entry a session, and there is no way to hand a setup to a
colleague. *10 · Setup file* saves it.

**One file, every well.** The active well's setup sits at the top level and
every other loaded well's per-well state goes in a section of its own, so a
library of five wells is one file rather than five. The shared settings — the
sample rate, the angles, the wavelet, the cutoffs — are stored once, never per
well, because five copies could disagree with each other and a comparison would
then be made on two sets of physics. A well the file carries but the session
has not loaded is **named and skipped**, not invented.

**It carries your interpretation, not your logs.** Names, depths, parameters
and choices — a couple of kilobytes of readable JSON with **no curve values in
it at all**, which is what makes it safe to keep beside the project, commit, or
mail. A test asserts that: every value of every curve in the well, checked
against every number in the file.

The consequence is deliberate — a setup cannot restore a session on its own.
Load the LAS, then apply the setup to it. The LAS stays the system of record;
the setup is what you decided about it.

Two things keep it trustworthy as the app grows:

- **Nothing is silently dropped.** `SAVED` and `EXCLUDED` between them name
  *every* field of `Settings`, and a test fails if a new one belongs to
  neither. A setup that quietly stopped carrying a field would restore most of
  a session and look complete, which is the worst way for this to fail. One
  field is excluded on purpose — an uploaded wavelet's samples, which are data
  rather than a decision.
- **It knows which well it came from.** Every file records the well's name,
  sample count, depth range and curve names — never curve values, which would
  both tie the file to one export and put a fingerprint of proprietary data in
  a shareable file. Applying a setup to a different well is then *reported and
  not prevented*: re-running a setup against a re-exported well is ordinary,
  and only the person doing it knows whether it is the same well.

### Where it goes

The download button hands the file to your browser, which puts it wherever its
Save dialog does. There is also a **write a copy to a folder** box, defaulting
to your Desktop — convenient, and worth understanding: it writes to the machine
*running the app*, which is your own only because the toolkit binds to
localhost. Deploy it for a team and that is the server's disk. The download is
the one that always does what it looks like.

### Putting it back on screen

Applying a setup restores the settings *and* the controls that show them. That
took two goes and is worth recording, because the obvious approaches both fail
silently. Streamlit gives a keyed widget's own state priority over the value it
is created with, and the browser re-sends that state on every rerun — so
writing the setting moves nothing, and clearing the widget's key server-side
does not help either. The control has to be **assigned** its new value. But a
widget's key cannot be assigned once the widget exists in that run, and the
setup section sits at the bottom of the page. So applying a setup stashes the
new control values and reruns, and they are written at the *top* of the next
run, ahead of the sidebar and every section below it.

Sections with their own *Apply* button get their inputs back; press each one to
recompute the curves from them.

### One trap worth recording

JSON has no integer keys. `zone_names` is keyed by the numeric code in the
`ZONE` curve, so a plain `json.dumps` turned `{1: "Shale"}` into
`{"1": "Shale"}` — after which every zone lookup by code missed, and a restored
session came back showing bare codes and seven intervals where the well has
four. The file looked perfectly correct. Dicts with non-string keys are now
written as an explicit list of pairs, and the "what changed" report compares
values as they really are rather than two JSON-safe copies, because comparing
those would call the broken and the correct version equal.

## A well with no shear sonic

Without Vs there is no gradient, no Vp/Vs, no Poisson, no LMR and no Gassmann.
The toolkit used to stop dead on a well that had none — which threw away the
well for the one curve that can honestly be predicted, and plenty of
exploration and older wells have no DTS at all. *5 · Shear sonic* on the Load
& QC page now predicts it. It sits **after** the petrophysics, because the best
predictor here is driven by VSH, and **before** the fluid cases, because
Gassmann needs Vs.

Four ways to get one, and the page **scores every one of them** against
whatever measured Vs the well does have rather than picking a default:

| Model | Knows about | On 15/9-19-A |
|---|---|---|
| Greenberg-Castagna, mixed by VSH | lithology | **4.4%**, unbiased, r 0.93 |
| this well's own Vp-Vs trend | this well, if the calibration is representative | 6.2% |
| Greenberg-Castagna, pure sandstone | nothing but Vp | 7.2%, **5.7% fast** |
| Castagna mudrock line | nothing but Vp | 7.7% |

Two of those numbers are worth dwelling on.

**The lithology-aware transform wins, and it is not close.** A Vp-Vs line has
nowhere to put shale volume, and shale volume is most of what moves Vs. Pure
sandstone reads 5.7% *fast* because it treats every shale as a sand — and a Vs
biased high pulls Vp/Vs down and moves every gradient with it.

**Fitting this well's own trend does not win**, which is the opposite of what
it feels like it should do. Blind-tested on a random half of 15/9-19-A the
fitted line reaches 6.2% against the transform's 4.2%. Worse, calibrated on the
shale-rich upper half and extrapolated into the sandier lower half — *exactly*
what you do when the shear sonic starts partway down the well — it collapses to
**11%, low by 11%**, because it is predicting shale velocities for sand. It is
a check and a fallback, not a default, and the scoreboard says so on screen.

### Where the provenance goes

A predicted Vs is named wherever the well is, not only on the page that
predicted it. The sidebar says so beside the well's name — *Vs predicted on
2 812 of 3 905 samples (72%) — every class on this well is provisional* — and
every reflector in the table carries `vs_predicted`, the share of **its own
lobe** whose Vs was invented rather than measured. An event blocked entirely
over predicted rock is a different kind of answer from one blocked over
measured rock, and nothing about A and B says so on its own.

That share is computed by carrying the per-sample mask on the well as an
ordinary 0/1 curve, so it resamples onto the time grid with everything else and
works for a well that is not the active one. It is deliberately not a canonical
curve, so the curve pickers — which enumerate `CANONICAL` — never offer it as
something to plot. Choosing *keep only what the file carries* removes it again.

### The part that is easy to miss

A prediction never overwrites a measurement — it fills gaps and nothing else,
and the mask that comes back is per sample, so "this curve is 72% invented" is
a statement the toolkit can actually make. But the honest warning is bigger
than that, and the page prints it:

> Blind-tested on 15/9-19-A — its real shear sonic removed, then predicted —
> the curve comes back within **4.4%** of the measurement and essentially
> unbiased. The reflector count still falls from **25 to 18**, and of the
> events found at the same depth in both runs **42% take a different AVO
> class**.

A gradient is a contrast attribute, and small velocity errors move contrasts
much more than they move velocities. 4.4% on the log is not 4.4% on the answer.
Every class on a predicted well is provisional, and `test_real_well.py` pins
that divergence so it cannot quietly improve into a claim nobody measured.

### Where another published transform goes

`polynomial_vs` is general: a `{lithology: polynomial}` map in km/s, Hill-averaged
over a mixture. Greenberg-Castagna is one parameterisation of it, and its
coefficients are **imported** from `core/rockphysics.py` rather than retyped, so
the trend lines the Rock Physics page draws and the curve predicted here cannot
come from two different sets of numbers. Another published set is a dictionary
entry in `TRANSFORMS`, not new code.

Only sets whose coefficients can be checked against their source belong there.
A transform carrying half-remembered numbers is worse than no transform,
because every Vp/Vs, Poisson, gradient and substitution downstream inherits
them with no mark on them.

## Where the classes are

The plainest question the classification can be asked, and the one that gets
lost among the statistics: **where in the well are the Class III events, and
what is the rock doing there?** No fitting, no ranking — the reflectors, where
they are.

It is drawn as small multiples rather than one crowded plot. Every panel shares
the depth axis, so a horizontal line across the figure is one reflector and the
panels can be read against each other by eye. Pick the depth reference and the
panels you want: intercept, gradient, fluid factor, φ, net-to-gross, VSH, SW,
Δφ across the reflector, amplitude — whichever the well actually carries.

Three choices in it worth naming, because each is the kind of thing that is
invisible when right and wrong in a way nobody reports:

- **The depth axis points down, and prefers TVDSS.** Depth increases down a
  well and must increase down the plot. Measured depth starts at a rig floor,
  so a class-against-depth plot drawn on it says where a class is in the
  *hole*; two wells cannot be read together on it at all.
- **The class is on the first panel's own axis, not only in the colour.** The
  class palette warns on contrast against a light surface, which obliges
  visible labels — and a figure whose only encoding is hue stops working when
  it is printed, photocopied, or read by someone who cannot separate the reds
  from the greens. Class order is fixed, never sorted by count, so one well
  looks the same on every rerun and two wells look like each other.
- **Colour is the class and shape is the well.** Two variables, two encodings.
  Using colour for both would make a Class III in one well look like a Class I
  in another.

On the **Multi-well** page the same figure runs over every compared well at
once. A class that clusters at one depth *across* wells is a rock property; one
that appears at a different depth in each is more likely a fluid or a facies
change. A well with no vertical reference is left out rather than drawn at the
wrong depth in the earth.

## AVO attributes

A and B are a coordinate system, not an answer. Everything on the **AVO
attributes** section is a way of reading a point in that plane against
something — a background trend, the mudrock line, or a rock property the well
actually measured.

### Derived per reflector

| Attribute | What it is |
|---|---|
| `rp` | P-impedance reflectivity — the intercept outright |
| `rs` | S-impedance reflectivity implied by A and B |
| `fluid_factor` | the P reflectivity the mudrock line does not explain |
| `ab_product` | A·B, the crude quick indicator |
| `vp_vs_background` | the ratio each of the above was computed at |

That last column is the one worth arguing about. `rs` comes out of the
two-term Aki-Richards form once a background Vp/Vs is assumed, and **at
Vp/Vs = 2 the density terms cancel identically** — which is exactly why every
textbook writes the relation as `Rs = (A − B)/2` there and nowhere else. Away
from 2 they do not cancel, and a fixed ratio quietly converts a density
contrast into shear reflectivity. So the ratio is read per reflector from its
own **upper blocked layer** — the incident medium, which is what the incidence
angle and therefore the gradient are measured in. On the demo well that ranges
from 1.61 in the sand to 2.00 in the shale, and using a constant 2.0 instead
moves `rs` by more than 0.005 on some reflectors, which `test_avo_attributes.py`
pins.

The **fluid factor** (Smith & Gidlow) is zero for a brine clastic on
Castagna's mudrock line — that is what it is for. But the honest version of
that sentence is longer, and the toolkit says the longer one: the
*reflectivity* form leaves a residual wherever the density steps,

```
dF = ½ (Δρ/ρ) (1 − 1.16 · Vs/Vp)
```

which at Vp/Vs = 2 is about 0.42 times the density contrast. **A fluid factor
is therefore not read against zero.** A compaction boundary has one. That is
why the panel ranks it against its own background rather than against an
absolute level, and why the tests pin the residual as a prediction rather than
asserting it away.

### Which reflectors are unusual?

Ranked, not thresholded, and scored in units of **this well's own scatter** —
a median absolute deviation of the deviations themselves. The raw distance
from a background trend is not comparable between wells: a hole with a wide
A–B cloud has every event looking anomalous, and 0.02 off the trend means
different things in different wells.

Two choices inside that:

- **The MAD, not a standard deviation.** A standard deviation is inflated by
  the very anomalies being looked for, so the strongest events would quietly
  raise the yardstick they are measured with.
- **About the trend, not about the median deviation.** The trend already *is*
  the reference — a reflector on it is by definition ordinary — and
  re-centring would move the zero somewhere the trend never put it. A cloud
  sitting entirely to one side of its trend is a cloud of somewhat unusual
  reflectors, and a centred score would call every one of them normal.

You can rank on the distance from the trend, on the fluid factor, or on A·B;
each is scored the same way, against its own spread.

On the **Multi-well** page the same ranking runs across every compared well,
with each well **measured against its own trend and scored in its own
scatter** — pooling the scaling would let the noisiest hole set the yardstick
and bury a quiet well's one genuine standout. The rank still spans the wells,
because *what should I look at first* is one question across a field.

### Which direction in the A–B plane is the fluid?

Any direction is an attribute: `A cos χ + B sin χ`. Some angles have names —
0° is the intercept, 90° the gradient, 45° the scaled Poisson reflectivity —
but the useful one is found rather than named. The **χ sweep** correlates every
rotation with a rock property measured on the same reflectors (the lobe-averaged
φ, VSH, SW or net-to-gross from the section above) and reports the angle that
wins, its sign, and how strongly.

This is the intercept–gradient counterpart of the EEI χ sweep on the crossplots
page, and the two ask the same question in different domains. Correlation is
**Spearman** by default: the relationship between a reflection attribute and a
rock property is monotonic far more often than it is linear, and a few dozen
reflectors is exactly the sample size where one outlier decides a Pearson
coefficient. The plot marks the trend's own perpendicular alongside the winner,
because where those two coincide the fluid direction and the anomaly direction
are the same thing — and where they do not, that is worth knowing.

With a few dozen reflectors the angle is soft. A broad flat peak is a *range*
of directions, and the caption says so rather than quoting a degree.

## What does the class depend on?

The classification says what the seismic does. This is the other half: does it
have anything to do with the rock? Is Class III where the porosity is, does
net-to-gross separate II from III, and is any of it just depth?

**The properties come from the same rock the class did.** Each reflector's φ,
VSH, SW and net-to-gross are averaged over the *same two half-lobes* its
intercept and gradient were fitted from — not read off the two samples at the
boundary. The class came from a lobe of rock; a porosity read from two samples
is describing something else, and setting one against the other compares two
different intervals.

Every property is reported three ways, and the third is the one to reach for:

| Column | What it is |
|---|---|
| `*_above`, `*_below` | the two half-lobes, in the wave's direction of travel |
| `d_*` | the change downwards across the reflector, `below − above` |
| `*_res` | whichever half-lobe carries **less shale** — the reservoir side |

The reservoir side exists because a sand *base* carries its reservoir above.
Plotting every class against "the porosity below" sets half the reflectors
against their seal and washes out whatever trend was there. The rule is
deliberately blunt and the choice it made is written into the table as
`reservoir_side`, so it can be read and argued with.

Net-to-gross is the mean of a 0/1 net flag over the same lobe, with one
deliberate refusal: a sample where the criterion cannot be evaluated is
**unknown, not non-net**. A gap in the VSH shortens the interval a
net-to-gross is measured over rather than quietly pushing it down. The cutoffs
live under **Net rock** in the sidebar and are the *only* definition of net in
the tool — the zone summary reads the same ones, because one screen carrying
two different net-to-grosses is worse than either.

### Ranked, and honest about it

The panel does not make you hunt through a selectbox. It runs a
**Kruskal-Wallis** test — one-way ANOVA on ranks, so nothing is assumed about
the shape of a porosity distribution within a class — on every property the
well offers, and ranks them by **effect size** (epsilon-squared: the fraction
of the property's rank variance the class label accounts for). Effect size and
not p, because with thirty reflectors the p-value mostly reports how many
events a property survives on.

Two caveats are built into what it prints, because both are ways this analysis
gets over-read:

- **`p (adj)` is Bonferroni** over the properties tested. Searching twenty
  properties for the one that separates best and then quoting its raw p as if
  you had asked once is how a coincidence becomes a finding. On 15/9-19-A the
  winner's raw p of 0.006 becomes 0.12 once the search is paid for — the tool
  says so rather than letting the small number stand.
- **Picked reflectors are not independent samples.** Neighbouring events see
  overlapping rock and one thick sand can produce several of them, so a well
  contributes far fewer independent observations than it does rows. Read the
  ranking as *which property separates the classes best*, not as a
  significance test.

### What it finds

On 15/9-19-A at the defaults, the ranking is unambiguous about one thing: the
**contrast** across a reflector separates the classes far better than either
side's absolute value.

| Property | ε² |
|---|---|
| Δφ across the reflector | 0.72 |
| ΔSw across the reflector | 0.57 |
| φ of the layer above | 0.13 |
| φ of the layer below | 0.07 |
| net-to-gross, reservoir side | 0.00 |
| depth | 0.00 |

Which is what the physics says — an intercept and a gradient are made of
contrasts, not of absolute properties — and it is worth seeing a tool derive
it from the data rather than assert it. The plot shows the same thing
directly: Class I events sit at negative Δφ, porosity dropping downwards into
a tighter layer, and Class IV at positive Δφ, porosity rising into a softer
one. That depth scores 0.00 is its own small relief: if class were just depth,
every property that varies with depth would look like a cause.

This is a measurement of one well at one set of defaults, not a law, and
`test_real_well.py` pins it so that a change to the blocking or the classifier
which overturns it shows up as a failing test rather than leaving a confident
sentence on screen with nothing behind it.

## More than one well

The toolkit held exactly one well for most of its life, and that shaped where
state lived: the tops, the rig-floor height, the water depth, the petrophysics
choice and the fluid model all sat on the shared settings, because there was
only ever one well for them to describe.

With a second well open that arrangement is a trap. Switching wells would leave
well B standing on well A's rig floor, filtered by zone names that do not exist
in it, and reporting a fluid case it never had — all of it silently, all of it
plausible-looking. So the state travels with the well:

- **Per well** — the fluid case and case mapping, zonation and zone names, tops,
  KB elevation and water depth, the deviation survey, the vertical-well
  declaration, the petrophysics choice and the fluid model. Loading a well puts
  its own state back; loading a *new* well starts it blank rather than
  inheriting the last one's.
- **Shared** — the sample rate, the time origin, the angle range, the wavelet,
  the reflectivity method, the amplitude cut and the class tolerance. These are
  how the wells are being *looked at*, not facts about any of them, and a
  comparison made on two different sets of physics compares the settings.

The **Active well** selector at the top of the sidebar switches which well
every other page works on. `Multi-well` is the exception: it runs on the wells
you tick, whichever one is active, each with its own per-well state, through
the same `reflector_analysis` the AVO Classification page uses — one pipeline,
so the two pages cannot come to disagree about the same reflector.

A well that cannot be analysed — too few angles configured, say — is **named
and skipped** rather than allowed to stop the others, and the analysis runs on
a button rather than on every rerun, because it is minutes of work on a real
well. Change a shared setting afterwards and the page says what is on screen is
stale instead of quietly redrawing half-old numbers.

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

**Where the naming does not follow the convention** — cases called `VP_1` and
`VP_2`, or an *oil* set that is really the in-situ one — the detector cannot
read it and has to be told. *5 · Fluid cases* on the Load & QC page shows each
case's three curves as dropdowns over the file's columns and applies the
assignment by hand.

### Modelling the cases a well does not have

Many wells arrive with one set of logs and nothing to compare it against. The
same section models the standard suite in one pass, and writes ordinary fluid
cases: brine, oil and gas, each labelled *(computed)*, flowing through every
page exactly as loaded ones do.

The fluids are **industry defaults you can overwrite** — seawater-salinity
brine, a medium live oil at 32° API and GOR 100, a slightly wet 0.65 gas — with
named presets for fresher and saltier brines, lighter and heavier oils, and
drier and richer gases. They are ordinary starting values, not this field's
fluids: a real salinity, API and GOR come from a PVT report. Pressure and
temperature are taken **from the depth log** through a hydrostatic gradient and
a geothermal one, so a substitution follows the conditions down the well rather
than holding one number over 600 m. Gas at 40 MPa is several times the modulus
of gas at 10, which is exactly what a fixed table gets wrong.

Three modelling choices are explicit rather than buried:

- **What is in the pores now.** A well logged in a gas leg is not brine-filled.
  Tell Gassmann it is and you are telling it the rock is stiffer than the logs
  say: on the demo well's gas sand every sample is refused with a negative dry
  frame. It does not quietly put the gas effect in twice, which is the failure
  mode worth having.
- **Water left in the hydrocarbon cases**, default 20%. A reservoir at residual
  water, not a pore of pure gas — which is not a rock that exists.
- **Reservoir only.** Gassmann assumes a connected, isotropic frame, and a
  shale is neither. Outside the VSH cutoff each case keeps the well's own
  curves rather than going blank, which is what an interpreter does and also
  what keeps the gather computable: a blank seal above a substituted sand
  deletes the very interface the case was built to show.

One result worth keeping in mind: **"brine, then oil, then gas" is a soft-rock
ordering, not a law.** Vp = √((K + 4µ/3)/ρ), so in a tight, stiff frame the
pore fluid barely moves K but still moves ρ, and gas can come out marginally
*faster* than oil. The demo well's cemented streak does exactly that, and the
tests pin it.

What the cases unlock:

- **Data & Crossplots** — overlay every case on the QI crossplots to see the
  fluid vector in AI–Vp/Vs space.
- **Synthetic Gather** — a gather per case, built on the shared time axis so
  the cases stay aligned sample for sample.
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

## Running it without the app

`avo_qi/core/` imports neither Streamlit nor Plotly — only numpy and scipy,
with pandas used just to assemble the reflector table — so the whole physics
layer is importable from a script or a notebook with no web stack loaded at
all. `avo_qi.analysis.reflector_analysis` is the one entry point the pages
themselves use, and it will run anywhere the two cached helpers it borrows
from `ui.py` can be reached.

There used to be a `python -m avo_qi.cli` command here. It was **removed
rather than repaired**, because it had silently fallen a refactor behind: it
still picked reflectors off a log-side reflectivity threshold with no blocking,
the method the app replaced. On 15/9-19-A, at identical settings, the command
reported **269 reflectors and 30 Class III** where the app reports **25 and 2**.
A headless path that disagrees with the application about what a reflector *is*
is worse than no headless path, and a second implementation of the same physics
was always going to drift again. Anything batch should call
`reflector_analysis` directly, so there is only ever one pipeline.

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

`test_well_library.py` and `test_multi_well.py` cover holding more than one
well. Most of what they check is provenance rather than arithmetic: that
switching wells takes each one's tops, datum and fluid case with it; that
dropping the active well falls back to whichever remains; and — on the
Multi-well page, running the demo well zoned by its `ZONE` curve beside the
North Sea well zoned by hand-entered tops — that each well's reflectors carry
**its own** formation names. A page that zoned every well by the active one's
tops would produce a table that looked entirely reasonable.

`test_properties.py` covers the per-reflector properties, mostly at the two
places they go quietly wrong: a net-to-gross must treat a gap in the VSH as a
*shorter interval*, not as non-net rock, and the reservoir side of a sand base
is the layer **above**. `test_real_well.py` adds two that only a real well can
make: that a curve is not reported where it was never logged — 15/9-19-A has
no VSH, PHI or SW over the top 166 m, and the resampler used to fill it with
each curve's first valid reading — and the measured ranking of what the class
depends on in that well, so a change to the blocking or the classifier that
overturns it fails a test rather than leaving a confident sentence on screen.

`test_project.py` covers the setup file, and two of its tests are the reason
the feature can be trusted: one asserts that **no curve value ever reaches the
file** — every sample of every curve checked against every number in it — and
one asserts that every field of `Settings` is either saved or explicitly
excluded, so a new setting cannot be silently forgotten by a file that still
looks complete.

`test_vs_prediction.py` and `TestAWellWithNoShearSonic` in `test_real_well.py`
cover predicting a shear sonic. The interesting ones are not the algebra: they
pin the *ranking* of the models measured on a real well — including that a
trend fitted to the well loses to a lithology-aware transform, and loses badly
when its calibration interval is a different rock from the prediction interval
— and they pin how far a predicted well diverges from the logged one it was
made from, so a 42% class disagreement cannot quietly become a claim that
prediction is harmless.

`test_avo_attributes.py` tests the derived attributes the strong way rather
than against their own formulae: it builds an interface from Vp, Vs and rho,
fits A and B off its **exact Zoeppritz** response, and asks whether the
pseudo-shear reflectivity comes back to the S-impedance reflectivity the logs
say it should be — it does, to 0.004. The fluid factor gets the same treatment
on two lower layers under one shale, one on the mudrock line and one softened
by gas: 0.003 against −0.107, a factor of thirty. Asserting the algebra
against itself would have passed with the coefficients transposed.

## References

- Aki & Richards (2002), *Quantitative Seismology*, 2nd ed. — reflectivity.
- Rutherford & Williams (1989); Castagna & Swan (1997) — AVO classes, Class IV
  and the A–B background trend.
- Shuey (1985) — the two-term `R(θ) = A + B sin²θ` intercept–gradient form.
- Whitcombe (2002) — extended elastic impedance, and the χ rotation the
  intercept–gradient sweep mirrors.
- Smith & Gidlow (1987), *Geophysical Prospecting* 35, 993–1014 — weighted
  stacking and the fluid factor. Fatti et al. (1994) — the reflectivity form
  it is computed in here.
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
