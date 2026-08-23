# AVO & QI Well Analysis Toolkit — Build Specification

A Streamlit application that takes a well **already containing Vp, Vs, and
RHOB**, produces the standard rock physics crossplots, generates a synthetic
**angle gather** from a user wavelet, and **classifies every reflector into
AVO classes** from an intercept–gradient analysis.

> **Amendment — Gassmann and Batzle-Wang are now in scope.** The original
> brief excluded them (see §1). That exclusion was lifted deliberately, so
> that the rock physics model could be driven per-sample from `VSH`, `PHIT`
> and `SW` — which needs a dry-frame-to-saturated step — and so that the well
> itself can be substituted rather than only read. The sections below are the
> original text; §1 carries the revised scope.

This document is the build brief. It gives the architecture, the physics
contracts (validated in prototype — reproduce them exactly), the UI
behaviour, and the acceptance tests. Streamlit is the end state; build and
test the `core/` physics first, headless.

---

## 1. Scope

**In:** Vp/Vs/RHOB logs → QI crossplots; angle-dependent reflectivity
(Zoeppritz + Aki-Richards) → wavelet convolution → synthetic angle gather;
per-reflector intercept A and gradient B → AVO class I–IV.

**In (added after the original brief):** a per-sample forward model driven by
`VSH`, `PHIT` and `SW`, with Gassmann as its saturation step
(`core/petro.py`, `core/gassmann.py`); Batzle-Wang fluid properties at
reservoir pressure and temperature (`core/fluids.py`); and Gassmann fluid
substitution of the loaded well, written back as ordinary fluid cases.

**Why the original exclusion was lifted.** The brief ruled out fluid
substitution, Batzle-Wang and Gassmann, on the basis that a well arrives with
substitution already done upstream. Reading a substituted well is still the
normal path, but two things could not be built without them:

* A model driven by the petrophysical logs has to get from a dry frame to a
  saturated rock, and that step *is* Gassmann. Without it the page could only
  compare a well against one hand-typed composition — which judges a shale
  against a quartz bound and says nothing.
* Substituting at the fixed room-condition constants in `FLUIDS` is wrong at
  depth: at 30 MPa and 90 °C a gas is roughly five times stiffer and six times
  denser than the table's entry. Batzle-Wang is what makes the substitution
  worth trusting, and it stays opt-in with the table as the default.

**Also added:** Monte Carlo uncertainty (`core/uncertainty.py`) over both the petrophysical logs and the frame
parameters, giving P10–P90 bands on the predicted logs and a *probability*
per AVO class rather than a single label. Off by default on every page: a
distribution is only as good as the spreads it was given, so it is never
produced without someone stating them.

**Still out:** anything that hides where a number came from. Cases computed
here are recorded in `WellData.computed_cases`, shown as *"(computed)"* in
every case selector, and warn before replacing a case that was loaded from the
file. A model of the well must never be mistakeable for a measurement of it.

---

## 2. Repository layout

```
avo_qi/
├── app.py                      # Streamlit entry + landing
├── SPEC.md                     # this document
├── requirements.txt
├── io/
│   └── loader.py               # LAS + CSV/Excel, mnemonic map, unit standardise
├── core/
│   ├── reflectivity.py         # Zoeppritz + Aki-Richards            [contract §4.1]
│   ├── wavelet.py              # Ricker + user wavelet ingest         [contract §4.2]
│   ├── synthetic.py            # RC series + gather assembly          [contract §4.3]
│   ├── avo.py                  # A/B fit (Shuey + Aki-R) + classifier [contract §4.4]
│   └── attributes.py           # AI, SI, Vp/Vs, Poisson, LMR, EEI     [contract §4.5]
├── pages/
│   ├── 1_Data_and_Crossplots.py
│   ├── 2_Synthetic_Gather.py
│   └── 3_AVO_Classification.py
├── sample_data/
│   └── demo_well.las           # synthetic 3-layer well for tests/demo
└── tests/
    └── test_core.py
```

---

## 3. Unit conventions (enforce in `io/` and UI, never in `core/`)

| Quantity   | Unit  | Convert on load                         |
|------------|-------|------------------------------------------|
| Velocity   | m/s   | ft/s → m/s; sonic µs/ft → m/s (1e6/DT)   |
| Density    | g/cc  | kg/m³ → /1000                            |
| Depth/time | TWT s | if only MD given, require a T-D or Vp for a checkshot-free integration |
| Angle      | deg   | core takes degrees, converts internally  |
| Frequency  | Hz    |                                          |

`core/` functions assume the table above. All conversion happens in
`io/loader.py` and the Streamlit widgets.

---

## 4. Physics contracts (prototype-validated — reproduce exactly)

The following behaviours were validated in a prototype and are the
regression targets. Reproduce them; do not weaken them.

### 4.1 `core/reflectivity.py`
Two functions, both vectorised over an array of incidence angles (degrees),
both returning Rpp for a single interface defined by
`(vp1,vs1,rho1, vp2,vs2,rho2)`:

- `zoeppritz_rpp(...)` — exact Zoeppritz P-P reflection, Aki & Richards
  convention. Ray parameter `p = sin(θ1)/vp1`; transmission and converted
  angles from Snell; clip `arcsin` arguments to [-1,1] to guard past
  critical angle.
- `aki_richards_rpp(...)` — 3-term linear approximation
  `R(θ) = ½(1−4(vs/vp)²sin²θ)·Δρ/ρ + ½·Δvp/vp·sec²θ − 4(vs/vp)²sin²θ·Δvs/vs`
  using averaged `vp,vs,rho` and contrasts across the interface.

**Validated cross-check:** for a soft gas-sand interface
(shale 2400/1200/2.35 over sand 2100/1300/2.10), the two agree to within
~0.005 in Rpp over 0–30° and diverge modestly by 40°; both are negative and
grow more negative with angle (Class III). Add this as a regression test.

Also provide a convenience `reflectivity_series(vp,vs,rho, angles, method)`
that walks a log and returns an `(n_samples × n_angles)` RC matrix (interface
`i` between sample `i` and `i+1`; last row zero).

### 4.2 `core/wavelet.py`
- `ricker(f, dt, length=0.128) -> (t, w)` — zero-phase Ricker; peak
  frequency `f` Hz, sample `dt` s. Normalise to unit peak amplitude.
- `load_wavelet(path_or_array, dt)` — ingest a user wavelet (CSV of
  amplitudes, or amplitude+time); resample to the trace `dt` if needed;
  report whether it is (near) zero-phase.
- `bandpass_ormsby(f1,f2,f3,f4, dt, length)` — optional alternative wavelet.

### 4.3 `core/synthetic.py`
- `build_gather(vp,vs,rho, angles, wavelet, dt, method) -> ndarray`
  `(n_samples × n_angles)`: for each angle, compute the RC series with the
  chosen reflectivity method, convolve with the wavelet (`mode='same'`),
  and stack columns into the gather. Depth axis is in TWT samples.
- `angle_stack(gather, angle_range)` — mean over an angle band (near/mid/far).
- `full_stack(gather)` — mean over all angles.

**Validated behaviour:** for a 3-layer shale/gas-sand/shale model with a
30 Hz Ricker and dt=1 ms, the sand top is a negative trough that brightens
(more negative) with angle, and the sand base mirrors it as a brightening
peak. Add this as a regression test.

### 4.4 `core/avo.py`
- `shuey_fit(R_of_theta, angles) -> (A, B)` — least-squares fit of
  `R = A + B·sin²θ` (2-term Shuey). `A` = intercept, `B` = gradient.
- `aki_richards_fit(R_of_theta, angles) -> (A, B[, C])` — fit the linearised
  form; return intercept and gradient (and optional third term) for
  comparison against Shuey.
- `classify(A, B, a_tol=0.02) -> str` — AVO class from the A–B pair:
  - **Class I**   : A > +a_tol, B < 0 (hard event, dims then reverses)
  - **Class IIp** : |A| ≤ a_tol, B < 0, A ≥ 0 (near-zero positive intercept)
  - **Class IIn** : |A| ≤ a_tol, B < 0, A < 0 (near-zero negative intercept,
    may show polarity flip)
  - **Class III** : A < −a_tol, B < 0 (classic bright gas sand)
  - **Class IV**  : A < 0, B > 0 (soft event, dims with offset)
  - else **background/other**
  Expose `a_tol` in the UI — the boundary between II and I/III is a tunable
  intercept band, not a hard sign split. (Prototype note: a marginal case at
  A=−0.037 sat on the II/III line; the tolerance band is what resolves it.)
- `reflector_avo(gather_or_rc, vp,vs,rho, angles, method, both=True)` —
  detect reflectors (interfaces above an amplitude/contrast threshold, or
  every non-trivial interface), fit A/B by **both** Shuey and Aki-Richards,
  classify, and return a table: depth/TWT, A_shuey, B_shuey, A_ar, B_ar,
  class, and Δ between the two fits.
- `background_trend(A, B)` — robust line through the A–B cloud for the
  fluid-anomaly overlay (deviation from background = candidate anomaly).

### 4.5 `core/attributes.py`
Pure array functions: `acoustic_impedance(vp,rho)`,
`shear_impedance(vs,rho)`, `vpvs(vp,vs)`, `poisson(vp,vs)`,
`lambda_rho(vp,vs,rho)`, `mu_rho(vs,rho)`,
`eei(vp,vs,rho, chi_deg, vp0,vs0,rho0)`.

---

## 5. Streamlit pages

Shared sidebar: uploaded well (cached in `st.session_state`), mnemonic
mapping, dt / TWT settings, angle range (min/max/step), wavelet chooser
(Ricker freq or upload), reflectivity method (Zoeppritz / Aki-Richards),
and the classifier `a_tol`.

1. **Data & Crossplots** — upload; mnemonic remap; log track display
   (Vp, Vs, RHOB, AI, Vp/Vs vs depth); the standard QI crossplots:
   AI vs Vp/Vs coloured by a third curve, λρ–μρ (LMR), IP–IS,
   Poisson vs AI; optional EEI with a χ-angle slider. All interactive
   (Plotly), depth- or facies-colourable.
2. **Synthetic Gather** — pick wavelet + angle range + method; build the
   angle gather; display it as a variable-density/wiggle panel beside the
   log tracks; show near/mid/far angle stacks and the full stack; export
   the gather (npy/segy-lite or csv).
3. **AVO Classification** — run `reflector_avo`; show the A–B crossplot with
   the class regions shaded and the background trend line; a reflector table
   (depth, A, B, class, Shuey-vs-AkiR Δ); pick a reflector to see its
   amplitude-vs-angle curve with both model fits overlaid; colour gather
   reflectors by assigned class.

---

## 6. Acceptance tests (`tests/test_core.py`, must pass under pytest)

1. **Zoeppritz vs Aki-Richards** — soft gas-sand interface: agreement to
   < 0.006 Rpp over 0–30°; both negative and monotonically more negative
   with angle to 40°.
2. **Zoeppritz normal incidence** — at θ=0, Rpp equals the acoustic
   impedance reflectivity `(AI2−AI1)/(AI2+AI1)`.
3. **Shuey fit recovers A,B** — synthetic `R = A + B sin²θ` with known A,B is
   recovered to < 1e-6.
4. **Classifier truth table** — hand-built interfaces for Classes I, IIp,
   IIn, III, IV each classify correctly at default `a_tol`.
5. **Gather signature** — 3-layer shale/gas-sand/shale, 30 Hz Ricker,
   dt=1 ms: sand top trough brightens (more negative) with angle; base peak
   brightens with angle.
6. **Wavelet** — Ricker is zero-phase (symmetric, peak at t=0), unit peak.
7. **Both-fit consistency** — for small angles (≤30°) Shuey and Aki-Richards
   A/B agree within a stated tolerance; the table reports Δ.

---

## 7. Dependencies (`requirements.txt`)

```
streamlit
numpy
scipy
pandas
lasio
plotly
openpyxl
segyio        # optional gather export
pytest        # dev
```

Prototype used only numpy/scipy for the physics — keep `core/` dependency-light
and free of Streamlit/plotting imports so it stays unit-testable.

---

## 8. Build order

1. `io/loader.py` + `sample_data/demo_well.las` generator (the 3-layer model).
2. `core/reflectivity.py`, `core/wavelet.py`, then tests 1–2, 6.
3. `core/synthetic.py` → test 5; `core/avo.py` → tests 3–4, 7;
   `core/attributes.py`.
4. `app.py` + pages 1→2→3, each smoke-tested against the demo well.
5. Full `pytest` green, then manual `streamlit run app.py`.

---

## 9. Reference conventions

- Reflectivity: Aki & Richards (2002), *Quantitative Seismology*, 2nd ed.
- AVO classes: Rutherford & Williams (1989), extended by Castagna & Swan
  (1997) for Class IV and the A–B background trend / fluid-anomaly view.
- Shuey (1985) two-term `R(θ) = A + B sin²θ` for the intercept–gradient fit.
