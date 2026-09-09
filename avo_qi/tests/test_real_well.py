"""End-to-end checks against a real well rather than the synthetic demo.

The bundled demo well is a clean three-layer model: 132 samples in time, every
curve complete, every boundary a step.  It is the right fixture for pinning a
known answer, and the wrong one for finding out what breaks on real data.
Every defect found in the blocking and reflector-picking work came from this
well and none of them from the demo — a reflector split on a neighbour's lobe,
a NaN polarity cast to ``INT_MIN``, and 42% of the reflector table describing
interfaces the seismic could not separate.

15/9-19-A is a North Sea well: 3905 samples over 3500–4095 m MD, with the gaps
real logs have — VSH on 2812 of 3905 samples, SW on 1965, RHOB carrying null
sentinels.  What it is for here is coverage of the messy path, so most of these
tests assert *invariants* rather than numbers.  The two that do pin numbers say
so, and exist to make a change in the defaults visible instead of silent.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pytest

streamlit = pytest.importorskip("streamlit")
pytest.importorskip("plotly")

from streamlit.testing.v1 import AppTest  # noqa: E402

from avo_qi.io.loader import read_well, standardise  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGES = os.path.join(HERE, "pages")
WELL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "data", "15_9_19_A.las")


def load():
    raw, units = read_well(WELL)
    return standardise(raw, units=units, name="15/9-19-A"), raw, units


def run_page(path, timeout=300):
    from avo_qi.ui import Settings

    well, raw, units = load()
    at = AppTest.from_file(path, default_timeout=timeout)
    at.session_state["well"] = well
    at.session_state["raw_df"] = raw
    at.session_state["raw_units"] = units
    at.session_state["settings"] = Settings()
    at.run()
    return at


def reflector_table(page):
    for element in page.dataframe:
        frame = element.value
        if hasattr(frame, "columns") and {"avo_class", "A_shuey"} <= set(frame.columns):
            return frame
    raise AssertionError("no reflector table on the page")


@pytest.fixture(scope="module")
def avo():
    return run_page(os.path.join(PAGES, "4_AVO_Classification.py"))


@pytest.fixture(scope="module")
def table(avo):
    assert not avo.exception
    return reflector_table(avo)


class TestTheWellLoads:
    def test_it_reads_with_its_gaps_intact(self):
        well, _, _ = load()
        df = well.df
        assert len(df) == 3905
        assert {"VP", "VS", "RHOB", "GR", "VSH", "SW", "DEPTH"} <= set(df.columns)

        # The null sentinel became NaN rather than surviving as -999.25, which
        # would sail through every finite check and poison the averages.
        for name in ("RHOB", "VSH", "SW"):
            values = df[name].to_numpy(float)
            assert not (values < -900).any(), f"{name} still carries a sentinel"
            assert np.isfinite(values).sum() < len(values), \
                f"{name} is complete here; this fixture is meant to have gaps"

    def test_the_sonics_are_physical(self):
        df = load()[0].df
        vp = df["VP"].to_numpy(float)
        vs = df["VS"].to_numpy(float)
        good = np.isfinite(vp) & np.isfinite(vs)
        assert (vs[good] < vp[good]).all()
        assert (vp[good] / vs[good] > np.sqrt(2)).all()


class TestEveryPageSurvivesIt:
    """The demo well is 132 samples of clean step boundaries. A page that only
    ever runs on that has not been run."""

    @pytest.mark.parametrize("page", [
        "1_Load_and_QC.py",
        "2_Data_and_Crossplots.py",
        "3_Synthetic_Gather.py",
        "4_AVO_Classification.py",
        "5_Rock_Physics.py",
    ])
    def test_it_runs_clean(self, page):
        at = run_page(os.path.join(PAGES, page))
        assert not at.exception, f"{page}: {at.exception}"


class TestTracePicking:
    """The invariant the AVO page rests on: the trace says where reflectors are."""

    @staticmethod
    def _full_stack(at):
        from avo_qi.core.synthetic import build_gather, full_stack
        from avo_qi.ui import build_wavelet, time_well

        settings = at.session_state["settings"]
        tw = time_well(at.session_state["well"], settings, settings.case)
        _, wavelet = build_wavelet(settings)
        return full_stack(build_gather(
            tw["VP"].to_numpy(float), tw["VS"].to_numpy(float),
            tw["RHOB"].to_numpy(float), settings.angles, wavelet,
            dt=settings.dt, method=settings.method))

    def test_every_reflector_is_a_turning_point(self, avo, table):
        from avo_qi.core.synthetic import _local_extrema

        stack = self._full_stack(avo)
        assert set(table["sample"].astype(int)) <= set(_local_extrema(stack).tolist())

    def test_the_polarity_reported_is_the_traces_own(self, avo, table):
        stack = self._full_stack(avo)
        for _, row in table.iterrows():
            expected = "peak" if stack[int(row["sample"])] > 0 else "trough"
            assert row["polarity"] == expected

    def test_every_event_gets_a_lobe_of_its_own(self, table):
        """This is the whole point of picking on the trace. Under log picking
        114 of 269 reflectors on this well had no turning point of their own
        and were blocked on a window belonging to a neighbour."""
        assert (table["blocking"] == "lobe").all()
        assert (table["lobe_samples"] > 0).all()

    def test_the_event_count_is_pinned(self, table):
        """A deliberate pin, not an invariant, and it says two things.

        The trace carries **25** events at the default 5% cut and a 30 Hz
        Ricker. The page shows **20** of them, because the interface-pair
        filter's default keeps only pairs whose lithology is known, and this
        well has no VSH over its top 166 m — five events sit in that interval
        and cannot be named. They used to pass the filter, labelled "silt over
        silt", only because the resampler filled that interval with the first
        VSH it could find (see
        ``TestACurveIsNotReportedWhereItWasNeverLogged``). The page says so on
        screen rather than dropping them quietly.

        If a default moves, this should fail rather than the numbers changing
        in silence.
        """
        from avo_qi.analysis import reflector_analysis
        from avo_qi.ui import Settings

        picked = reflector_analysis(load()[0], Settings())["table"]
        assert len(picked) == 25
        assert len(table) == 20
        unnamed = picked["litho_pair"].astype(str).str.contains("undefined")
        assert int(unnamed.sum()) == 5

    def test_the_seismic_merges_many_log_interfaces_into_each_event(self, avo, table):
        """3905 log samples over ~316 samples of two-way time cannot yield 269
        separable events at 30 Hz, which is what a log-side threshold claimed."""
        assert len(table) < 40
        span = table["lobe_samples"].to_numpy(float)
        assert span.min() >= 2
        assert np.median(span) > 8      # a real lobe, not a couple of samples


class TestNoSilentNumericalDamage:
    def test_the_pipeline_raises_no_runtime_warnings(self):
        """A NaN cast through ``np.sign(...).astype(int)`` lands on INT_MIN and
        matches nothing, with a RuntimeWarning as the only clue. Gaps in this
        well's curves are what make that reachable."""
        from avo_qi.core.blocking import (blocked_reflectivity,
                                          half_cycle_samples, lobe_windows)
        from avo_qi.core.reflectivity import reflectivity_series
        from avo_qi.core.synthetic import build_gather, full_stack, trace_events
        from avo_qi.core.tuning import apparent_period
        from avo_qi.ui import Settings, build_wavelet, time_well

        settings = Settings()
        tw = time_well(load()[0], settings, settings.case)
        vp = tw["VP"].to_numpy(float)
        vs = tw["VS"].to_numpy(float)
        rho = tw["RHOB"].to_numpy(float)
        _, wavelet = build_wavelet(settings)
        window = half_cycle_samples(apparent_period(wavelet, settings.dt), settings.dt)

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            stack = full_stack(build_gather(vp, vs, rho, settings.angles, wavelet,
                                            dt=settings.dt, method=settings.method))
            events = trace_events(stack, relative=settings.threshold)
            bounds = lobe_windows(stack, events["index"],
                                  polarity=events["polarity"].astype(float),
                                  max_half_width=2 * window)
            blocked = blocked_reflectivity(
                vp, vs, rho, events["index"], settings.angles, window=window,
                method="backus", guard=2, reflectivity_method=settings.method,
                bounds=bounds)
            reflectivity_series(vp, vs, rho, settings.angles, method=settings.method)

        for key in ("vp_upper", "vs_upper", "rho_upper",
                    "vp_lower", "vs_lower", "rho_lower"):
            assert np.isfinite(blocked[key]).all(), f"{key} came out non-finite"

    def test_the_fitted_intercepts_and_gradients_are_all_finite(self, table):
        for column in ("A_shuey", "B_shuey", "A_ar", "B_ar"):
            assert np.isfinite(table[column].to_numpy(float)).all()

    def test_shuey_and_aki_richards_agree(self, table):
        """Two independent fits to the same layers. A real well is where a
        disagreement would show up if the fit were being fed rubbish."""
        assert np.abs(table[["dA", "dB"]].to_numpy(float)).max() < 1e-6


class TestZonationFromTops:
    """15/9-19-A has no ZONE curve, so without tops the whole zone machinery —
    the filter, the boundary flag, the per-zone summaries — is dead on it.

    The depths below are invented for the test, not this field's real
    stratigraphy; what is being checked is the wiring, not the geology.
    """

    TOPS = [{"zone": "Upper", "top": 3500.0},
            {"zone": "Middle", "top": 3700.0},
            {"zone": "Reservoir", "top": 3900.0}]

    def test_the_well_has_no_zone_curve_to_begin_with(self):
        assert "ZONE" not in load()[0].df.columns

    def test_tops_produce_intervals_closed_at_the_bottom_of_the_well(self):
        from avo_qi.ui import Settings, well_zones

        well = load()[0]
        settings = Settings()
        assert well_zones(well, settings) is None      # nothing without tops

        settings.zone_tops = list(self.TOPS)
        intervals = well_zones(well, settings)
        assert list(intervals["zone"]) == ["Upper", "Middle", "Reservoir"]
        assert list(intervals["top"]) == [3500.0, 3700.0, 3900.0]
        # Each runs to the next, and the deepest is closed just *past* the last
        # sample rather than on it: assignment is `top <= depth < base`, so a
        # base exactly at total depth leaves the bottom sample unzoned.
        assert list(intervals["base"])[:2] == [3700.0, 3900.0]
        deepest = float(well.df["DEPTH"].max())
        step = float(np.median(np.diff(np.sort(well.df["DEPTH"].to_numpy(float)))))
        assert intervals["base"].iloc[-1] == pytest.approx(deepest + step)
        assert intervals["base"].iloc[-1] > deepest

    def test_every_sample_gets_a_zone(self):
        from avo_qi.core.zones import UNZONED
        from avo_qi.ui import Settings, zone_labels

        well = load()[0]
        settings = Settings()
        settings.zone_tops = list(self.TOPS)
        labels = zone_labels(well.df, well, settings)
        assert set(labels) == {"Upper", "Middle", "Reservoir"}
        assert UNZONED not in set(labels)

    def test_the_avo_page_carries_a_zone_per_event(self):
        at = run_page(os.path.join(PAGES, "4_AVO_Classification.py"))
        assert set(reflector_table(at)["zone"]) == {"unzoned"}

        at.session_state["settings"].zone_tops = list(self.TOPS)
        at.run()
        assert not at.exception
        table = reflector_table(at)
        assert set(table["zone"]) <= {"Upper", "Middle", "Reservoir"}
        assert len(set(table["zone"])) > 1

    def test_events_on_a_top_are_flagged_as_zone_boundaries(self):
        at = run_page(os.path.join(PAGES, "4_AVO_Classification.py"))
        at.session_state["settings"].zone_tops = list(self.TOPS)
        at.run()
        table = reflector_table(at)
        flagged = table[table["is_zone_boundary"]]
        assert len(flagged), "no event straddles a top"
        # A flagged event has different zones above and below it.
        for _, row in flagged.iterrows():
            assert row["zone"] != row["zone_below"]

    def test_a_top_at_the_very_start_of_the_well_still_labels_samples(self):
        """The warning on the load page tested the *top* against the well's
        range and called this empty: the shallowest top is 3500.0 while the
        first sample is at 3500.0183, and `depth >= top` matches it fine."""
        from avo_qi.core.zones import assign_zones
        from avo_qi.ui import Settings, well_zones

        well = load()[0]
        settings = Settings()
        settings.zone_tops = list(self.TOPS)
        intervals = well_zones(well, settings)

        depth = well.df["DEPTH"].to_numpy(float)
        assert depth.min() > 3500.0, "the fixture must start below the top"
        labels = assign_zones(depth, intervals)
        assert (labels == "Upper").sum() > 0

    def test_the_zone_filter_narrows_the_events(self):
        at = run_page(os.path.join(PAGES, "4_AVO_Classification.py"))
        at.session_state["settings"].zone_tops = list(self.TOPS)
        at.run()
        everything = len(reflector_table(at))

        at.session_state["settings"].zones = ["Reservoir"]
        at.run()
        assert not at.exception
        narrowed = reflector_table(at)
        assert 0 < len(narrowed) < everything
        # Kept when *either* side is in the zone, so the top of the interval
        # survives along with everything inside it.
        assert {"Reservoir"} <= set(narrowed["zone"]) | set(narrowed["zone_below"])

    def test_tops_do_not_leak_into_the_next_well(self):
        """The same failure the zone *selection* had once: a top is a depth in
        the well it was typed against and means nothing in another.

        Driven by actually loading a second well through the page's own button,
        so it is `set_well` doing the clearing and not the test.
        """
        at = run_page(os.path.join(PAGES, "1_Load_and_QC.py"))
        at.session_state["settings"].zone_tops = list(self.TOPS)
        at.run()
        assert at.session_state["settings"].zone_tops
        assert at.session_state["well"].name == "15/9-19-A"

        next(b for b in at.button if b.label == "Load demo well").click().run()
        assert not at.exception
        assert at.session_state["well"].name == "DEMO-1"
        assert at.session_state["settings"].zone_tops == []


class TestRockPhysicsTemplateOnARealWell:
    """A template is only as relevant as the lithology it was drawn for."""

    PHI = np.linspace(0.02, 0.36, 25)

    def _median(self):
        from avo_qi.core.attributes import acoustic_impedance, vpvs

        df = load()[0].df
        good = (np.isfinite(df["VP"]) & np.isfinite(df["VS"])
                & np.isfinite(df["RHOB"])).to_numpy()
        return (float(np.median(acoustic_impedance(df["VP"][good], df["RHOB"][good]))),
                float(np.median(vpvs(df["VP"][good], df["VS"][good]))),
                float(np.nanmedian(df["VSH"])))

    def test_drawing_it_at_the_wells_own_vsh_moves_it_onto_the_data(self):
        """The clean-sand template misses this well by a wide margin in Vp/Vs,
        and drawing it at the well's median VSH closes most of that gap. If it
        did not, the shale fraction would not be reaching the mineral mix."""
        from avo_qi.core.petro import rock_physics_template

        ai_med, vpvs_med, vsh_med = self._median()
        assert 0.2 < vsh_med < 0.5, "this well should be middling shaly"

        def gap(vsh):
            template = rock_physics_template(self.PHI, [1.0], vsh=vsh)
            k = int(np.nanargmin(np.abs(template["AI"][:, 0] - ai_med)))
            return abs(template["vpvs"][k, 0] - vpvs_med)

        clean = gap(0.0)
        own = gap(vsh_med)
        assert own < clean / 3.0, f"clean {clean:.3f} vs own {own:.3f}"
        assert own < 0.05

    def test_the_gas_line_sits_below_the_brine_line_throughout(self):
        from avo_qi.core.petro import rock_physics_template

        _, _, vsh_med = self._median()
        template = rock_physics_template(self.PHI, [1.0, 0.0], vsh=vsh_med)
        assert np.all(template["vpvs"][:, 1] < template["vpvs"][:, 0])
        assert np.all(template["AI"][:, 1] < template["AI"][:, 0])


class TestLithologyOnRealCurves:
    def test_pairs_come_from_the_wells_own_vsh(self, table):
        from avo_qi.core.lithology import LITHOLOGIES

        pairs = set(table["litho_pair"])
        assert pairs, "no lithology pairs at all"
        # Real VSH here spans 0.02–0.80, so more than one class must appear.
        named = {p for p in pairs if "undefined" not in p}
        assert len(named) > 1
        for pair in named:
            upper, lower = pair.split(" over ")
            assert upper in LITHOLOGIES and lower in LITHOLOGIES

    def test_the_pair_filter_narrows_the_table(self):
        # Its own page run: setting a widget mutates the AppTest, and the
        # module-scoped `avo` fixture is shared with every test above.
        at = run_page(os.path.join(PAGES, "4_AVO_Classification.py"))
        picker = next(m for m in at.multiselect
                      if m.label == "Interface pairs to keep")
        assert len(picker.options) > 1
        target = picker.options[0]
        picker.set_value([target]).run()
        assert not at.exception
        assert set(reflector_table(at)["litho_pair"]) == {target}


class TestZoneSummaryOnARealWell:
    """The summary on a well with the gaps real logs have.

    15/9-19-A logs VSH on 2812 of 3905 samples and SW on 1965, so this is where
    a net-to-gross quoted without its coverage would be quietly wrong.  The
    tops are invented, as above; the wiring is what is under test.
    """

    TOPS = TestZonationFromTops.TOPS

    @staticmethod
    def _summary(page):
        """The widest zone-summary frame on the page.

        The page shows the answer columns first and every column behind an
        expander, so there are two; the tests want the complete one.
        """
        found = [e.value for e in page.dataframe
                 if hasattr(e.value, "columns")
                 and {"zone", "gross", "ntg"} <= set(e.value.columns)]
        if not found:
            raise AssertionError("no zone summary on the page")
        return max(found, key=lambda f: len(f.columns))

    @pytest.fixture(scope="class")
    @classmethod
    def zoned(cls):
        at = run_page(os.path.join(PAGES, "4_AVO_Classification.py"))
        at.session_state["settings"].zone_tops = list(cls.TOPS)
        at.run()
        assert not at.exception
        return at

    def test_there_is_no_summary_until_there_are_tops(self, avo):
        with pytest.raises(AssertionError):
            self._summary(avo)

    def test_the_thicknesses_add_up_to_the_logged_interval(self, zoned):
        summary = self._summary(zoned)
        assert list(summary["zone"]) == ["Upper", "Middle", "Reservoir"]
        depth = load()[0].df["DEPTH"].to_numpy(float)
        assert summary["gross"].sum() == pytest.approx(depth.max() - depth.min(),
                                                       abs=0.5)

    def test_net_never_exceeds_gross_and_pay_never_exceeds_net(self, zoned):
        summary = self._summary(zoned)
        assert (summary["net"] <= summary["gross"] + 1e-9).all()
        assert (summary["pay"] <= summary["net"] + 1e-9).all()
        assert summary["ntg"].between(0.0, 1.0).all()

    def test_a_gap_in_a_curve_shows_up_as_coverage_not_as_zero_net(self, zoned):
        """VSH is missing over the shallow part of this well.  That has to read
        as *not judged* rather than as *not reservoir*."""
        summary = self._summary(zoned).set_index("zone")
        assert summary["net_coverage"].min() < 0.95
        # ...and where a curve is missing entirely the cutoff cannot bite: the
        # zone still reports its gross thickness and its logged averages.
        assert (summary["gross"] > 0).all()
        assert summary["mean_VSH"].notna().any()

    def test_the_events_are_attributed_to_zones(self, zoned):
        summary = self._summary(zoned)
        table = reflector_table(zoned)
        assert summary["n_events"].sum() >= len(table)   # boundaries count twice
        assert summary["class_mix"].notna().any()


class TestDepthReferenceOnARealWell:
    """15/9-19-A carries no TVD curve, and its EKB, EGL, KB and GL header
    entries are all empty — which is the normal case, and why the toolkit asks
    instead of reading. The datum below is invented for the test.
    """

    KB = 25.0
    WATER = 90.0

    def test_the_file_offers_nothing_to_start_from(self):
        from avo_qi.io.loader import read_las_header

        well = load()[0]
        assert not {"TVD", "TVDSS", "TVDBML"} & set(well.df.columns)
        assert read_las_header(WELL) == {"kb_elevation": None,
                                         "ground_level": None,
                                         "water_depth": None}

    @staticmethod
    def _zoned(vertical=True, survey=None):
        """The AVO page on a well whose depth reference has been resolved."""
        from avo_qi.core.depth import depth_references

        well, raw, units = load()
        resolved = depth_references(
            well.df["DEPTH"].to_numpy(float), vertical=vertical, survey=survey,
            kb_elevation=TestDepthReferenceOnARealWell.KB,
            water_depth=TestDepthReferenceOnARealWell.WATER)
        for name in ("TVD", "TVDSS", "TVDBML"):
            well.df[name] = resolved[name]

        at = AppTest.from_file(os.path.join(PAGES, "4_AVO_Classification.py"),
                               default_timeout=300)
        at.session_state["well"] = well
        at.session_state["raw_df"] = raw
        at.session_state["raw_units"] = units
        from avo_qi.ui import Settings

        at.session_state["settings"] = Settings()
        at.run()
        assert not at.exception
        return at

    def test_every_event_carries_its_true_vertical_depth(self):
        table = reflector_table(self._zoned())
        for column in ("depth", "tvd", "tvdss", "tvdbml"):
            assert column in table.columns, column
        md = table["depth"].to_numpy(float)
        assert table["tvdss"].to_numpy(float) == pytest.approx(md - self.KB,
                                                               abs=0.01)
        assert table["tvdbml"].to_numpy(float) == pytest.approx(
            md - self.KB - self.WATER, abs=0.01)

    def test_a_deviated_hole_moves_every_event_up(self):
        """The reason MD will not do: this well's events are hundreds of metres
        shallower in TVD once the hole is deviated, and an AVO-class-against-
        depth trend built on MD would put them all too deep."""
        survey = ([0.0, 1000.0, 4200.0], [0.0, 0.0, 50.0], [0.0, 120.0, 120.0])
        straight = reflector_table(self._zoned())
        bent = reflector_table(self._zoned(vertical=False, survey=survey))
        assert (bent["tvdss"].to_numpy() < straight["tvdss"].to_numpy()).all()
        assert (straight["tvdss"] - bent["tvdss"]).min() > 100.0

    def test_the_references_survive_the_trip_through_the_time_axis(self):
        """They are interpolated onto the time grid with every other curve, so
        an event's TVDSS is read at the same sample as its amplitude."""
        table = reflector_table(self._zoned())
        assert (np.diff(table["tvdss"].to_numpy(float)) > 0).all()
        assert table["tvdss"].notna().all()


class TestThePetrophysicsQuestionOnARealWell:
    """15/9-19-A carries both the raw logs and a full interpretation — VSH,
    PHIF, SW — and also the parameter curves that interpretation was made with:
    RHOMA, RHOFL, RW, M, N, GRMIN, GRMAX. That makes it the well that shows why
    the toolkit asks rather than assumes, and why the defaults come off the
    file rather than out of a textbook.
    """

    @staticmethod
    def _page():
        return run_page(os.path.join(PAGES, "1_Load_and_QC.py"))

    @staticmethod
    def _compute(at, mode="all"):
        next(r for r in at.radio
             if r.label.startswith("Where VSH")).set_value(mode).run()
        next(b for b in at.button
             if b.label == "Apply interpretation").click().run()
        assert not at.exception
        return at

    def test_the_defaults_come_from_the_wells_own_parameter_curves(self):
        at = self._page()
        next(r for r in at.radio
             if r.label.startswith("Where VSH")).set_value("all").run()
        values = {n.label: n.value for n in at.number_input}
        assert values["Matrix density (g/cc)"] == pytest.approx(2.66)   # RHOMA
        assert values["Fluid density (g/cc)"] == pytest.approx(0.80)    # RHOFL
        assert values["Rw (ohm·m)"] == pytest.approx(0.0211, abs=1e-3)  # RW
        assert values["Cementation m"] == pytest.approx(1.793, abs=1e-3)  # M
        assert values["Saturation n"] == pytest.approx(2.45, abs=1e-3)    # N
        assert values["GR clean (API)"] == pytest.approx(14.0)          # GRMIN
        assert values["GR shale (API)"] == pytest.approx(115.0)         # GRMAX
        assert any("parameter curves" in c.value for c in at.caption)

    def test_computing_with_them_reproduces_the_files_interpretation(self):
        """Not a coincidence and not a tautology: the transforms here are the
        standard ones, so given the same parameters they land on the same
        answer. Porosity comes back to within 0.0002 in the median."""
        at = self._compute(self._page())
        well = at.session_state["well"]
        original = at.session_state["petro_original"]
        for curve, correlation, difference in (("PHI", 0.90, 0.01),
                                               ("SW", 0.90, 0.02),
                                               ("VSH", 0.90, 0.05)):
            mine = well.df[curve].to_numpy(float)
            theirs = np.asarray(original[curve], dtype=float)
            both = np.isfinite(mine) & np.isfinite(theirs)
            assert both.sum() > 1000, curve
            assert np.corrcoef(mine[both], theirs[both])[0, 1] > correlation, curve
            assert abs(np.median(mine[both] - theirs[both])) < difference, curve

    def test_the_textbook_defaults_would_have_been_worse(self):
        """Which is the argument for reading the file's parameters at all."""
        from avo_qi.core.petrophysics import porosity_from_density

        well = load()[0]
        rhob = well.df["RHOB"].to_numpy(float)
        theirs = well.df["PHI"].to_numpy(float)
        both = np.isfinite(rhob) & np.isfinite(theirs)
        textbook = porosity_from_density(rhob, 2.65, 1.00)["phi"]
        from_file = porosity_from_density(rhob, 2.66, 0.80)["phi"]
        assert abs(np.median(from_file[both] - theirs[both])) < \
            abs(np.median(textbook[both] - theirs[both]))

    def test_filling_only_the_gaps_leaves_the_file_curves_alone(self):
        """This well has all three, so 'fill' must change nothing."""
        at = self._page()
        before = {c: at.session_state["well"].df[c].to_numpy(float).copy()
                  for c in ("VSH", "PHI", "SW")}
        self._compute(at, mode="fill")
        for curve, values in before.items():
            assert np.allclose(at.session_state["well"].df[curve].to_numpy(float),
                               values, equal_nan=True), curve
        assert set(at.session_state["petro_source"].values()) == {"file"}


class TestModellingFluidCasesOnARealWell:
    """15/9-19-A carries one set of logs and no fluid cases at all, which is
    the ordinary case and the reason the toolkit models them."""

    def test_it_arrives_with_nothing_to_compare(self):
        well = load()[0]
        assert well.cases == ["in situ"]
        assert not well.has_fluid_cases

    @staticmethod
    def _modelled():
        at = run_page(os.path.join(PAGES, "1_Load_and_QC.py"))
        next(b for b in at.button
             if b.label == "Model the fluid cases").click().run()
        assert not at.exception
        return at

    def test_the_three_cases_appear_and_are_labelled_computed(self):
        well = self._modelled().session_state["well"]
        assert well.cases == ["in situ", "brine", "oil", "gas"]
        assert well.computed_cases == ["brine", "oil", "gas"]

    def test_the_reservoir_moves_and_the_rest_of_the_well_does_not(self):
        well = self._modelled().session_state["well"]
        # The page's own criterion, not a variant of it: VSH at or below the
        # silty-sand cutoff, wherever VSH exists.
        vsh = well.df["VSH"].to_numpy(float)
        reservoir = np.isfinite(vsh) & (vsh <= 0.35)
        assert reservoir.sum() > 300

        in_situ = well.frame("in situ")["VP"].to_numpy(float)
        gas = well.frame("gas")["VP"].to_numpy(float)
        assert not np.allclose(gas[reservoir], in_situ[reservoir])
        assert np.allclose(gas[~reservoir], in_situ[~reservoir], equal_nan=True)

    def test_the_fluids_come_out_in_their_physical_order(self):
        well = self._modelled().session_state["well"]
        moved = ~np.isclose(well.frame("gas")["VP"].to_numpy(float),
                            well.frame("in situ")["VP"].to_numpy(float))
        assert moved.sum() > 300
        rho = {c: well.frame(c)["RHOB"].to_numpy(float)[moved]
               for c in ("brine", "oil", "gas")}
        assert (rho["brine"] > rho["oil"]).all() and (rho["oil"] > rho["gas"]).all()
        vp = {c: well.frame(c)["VP"].to_numpy(float)[moved]
              for c in ("brine", "oil", "gas")}
        assert np.median(vp["brine"]) > np.median(vp["oil"]) > np.median(vp["gas"])

    def test_a_modelled_case_carries_through_to_the_avo_page(self):
        """The point of writing them as ordinary cases: every other page picks
        them up through machinery that already existed."""
        at = self._modelled()
        well = at.session_state["well"]
        settings = at.session_state["settings"]
        settings.case = "gas"

        page = AppTest.from_file(os.path.join(PAGES, "4_AVO_Classification.py"),
                                 default_timeout=300)
        page.session_state["well"] = well
        page.session_state["raw_df"] = at.session_state["raw_df"]
        page.session_state["raw_units"] = at.session_state["raw_units"]
        page.session_state["settings"] = settings
        page.run()
        assert not page.exception
        assert len(reflector_table(page)) > 0


class TestACurveIsNotReportedWhereItWasNeverLogged:
    """The resampler used to extrapolate, and everything downstream believed it.

    15/9-19-A carries no VSH, PHI or SW above 3666 m — 166 m of the logged
    interval, a quarter of the well. ``np.interp`` clamps outside its data
    range, so the time frame came back with *zero* missing samples: a constant
    VSH of 0.599, a constant porosity of 0.200 and a constant SW of 0.559
    filling rock nobody had interpreted. Nothing about that reads as wrong
    downstream — the lithology pair said "silt over silt" with conviction, and
    a net-to-gross could be measured over an interval with no petrophysics in
    it at all. This is the fixture that shows it, because the demo well is
    complete and never could.
    """

    @staticmethod
    def timed():
        from avo_qi.ui import Settings, time_well

        return time_well(load()[0], Settings())

    @pytest.mark.parametrize("curve", ["VSH", "PHI", "SW"])
    def test_the_unlogged_section_comes_back_missing(self, curve):
        frame = self.timed()
        values = frame[curve].to_numpy(float)
        assert not np.isfinite(values).all(), curve
        # Missing at the top, where the curve does not start until 3666 m.
        assert not np.isfinite(values[0])

    @pytest.mark.parametrize("curve", ["VSH", "PHI", "SW"])
    def test_nothing_survives_outside_the_curves_own_interval(self, curve):
        """Every finite time sample has to sit inside the depth range the
        curve was actually logged over."""
        well = load()[0]
        depth = well.df["DEPTH"].to_numpy(float)
        logged = np.isfinite(well.df[curve].to_numpy(float))
        first, last = depth[logged].min(), depth[logged].max()

        frame = self.timed()
        good = np.isfinite(frame[curve].to_numpy(float))
        seen = frame["DEPTH"].to_numpy(float)[good]
        step = float(np.median(np.diff(depth)))
        assert seen.min() >= first - step
        assert seen.max() <= last + step

    @pytest.mark.parametrize("curve", ["VP", "VS", "RHOB"])
    def test_the_elastic_logs_are_untouched(self, curve):
        """The fix must not put holes in the three curves the whole synthetic
        is built from — they are complete over the timed interval, and a NaN
        in any of them would blank the trace."""
        assert np.isfinite(self.timed()[curve].to_numpy(float)).all()

    def test_the_reflectors_in_that_section_report_no_lithology(self):
        """Which is the honest answer, and was 'silt over silt' before."""
        from avo_qi.analysis import reflector_analysis
        from avo_qi.ui import Settings

        table = reflector_analysis(load()[0], Settings())["table"]
        shallow = table[table["depth"] < 3666.0]
        assert len(shallow) >= 3
        assert (shallow["litho_pair"] == "undefined over undefined").all()
        assert shallow["ntg_below"].isna().all()
        assert shallow["phi_res"].isna().all()


class TestWhatTheClassDependsOnInThisWell:
    """The class-against-property panel, measured on a real well.

    The caption on that panel makes a claim — that the *contrast* across a
    reflector separates the classes better than either side's absolute value,
    because an intercept and a gradient are made of contrasts. This is the
    measurement behind it. It is a fact about 15/9-19-A at the default
    settings, not a law, and it is pinned so that a change in the blocking or
    the classifier that overturns it shows up here rather than leaving a
    confident sentence on screen with nothing behind it.
    """

    @staticmethod
    def ranked():
        from avo_qi.ui import Settings, property_options
        from avo_qi.analysis import reflector_analysis
        from avo_qi.ui import class_dependence_ranking

        table = reflector_analysis(load()[0], Settings())["table"]
        return table, class_dependence_ranking(table, property_options(table))

    def test_the_porosity_contrast_beats_either_side_alone(self):
        _, ranking = self.ranked()
        effect = ranking.set_index("column")["eps2"]
        assert effect["d_phi"] > effect["phi_above"]
        assert effect["d_phi"] > effect["phi_below"]
        assert effect["d_phi"] > 0.3          # a strong separation, not a nudge

    def test_the_best_property_does_not_survive_the_search_correction(self):
        """Seventeen properties tested, so the winner's raw p of 0.006 is worth
        about 0.1 once the search is paid for. The panel has to say so — this
        is the number that stops a coincidence being reported as a finding."""
        _, ranking = self.ranked()
        best = ranking.dropna(subset=["p"]).iloc[0]
        assert best["p"] < 0.05
        assert best["p_adj"] > 0.05

    def test_depth_alone_does_not_sort_the_classes(self):
        """Worth knowing before reading anything into the rest: if class were
        just depth, every property that varies with depth would look like a
        cause."""
        _, ranking = self.ranked()
        effect = ranking.set_index("column")["eps2"]
        assert effect["depth"] < 0.06         # below Cohen's 'moderate'


class TestAWellWithNoShearSonic:
    """15/9-19-A with its DTS removed — the case the toolkit used to reject.

    A missing shear sonic stopped the Load & QC page dead, which threw away a
    well for the one curve that can honestly be predicted. These tests hold
    both halves of the fix: that such a well now runs end to end, and that the
    result is *not* quietly treated as equivalent to a logged one.
    """

    @staticmethod
    def stripped():
        well, raw, units = load()
        truth = well.df["VS"].to_numpy(float).copy()
        well.df = well.df.drop(columns=["VS"])
        return well, raw, units, truth

    @staticmethod
    def page(well, raw, units):
        from avo_qi.ui import Settings

        at = AppTest.from_file(os.path.join(PAGES, "1_Load_and_QC.py"),
                               default_timeout=300)
        at.session_state["well"] = well
        at.session_state["raw_df"] = raw
        at.session_state["raw_units"] = units
        at.session_state["settings"] = Settings()
        at.run()
        return at

    def test_the_page_no_longer_stops_dead(self):
        well, raw, units, _ = self.stripped()
        at = self.page(well, raw, units)
        assert not at.exception
        # Every step is reachable; it used to stop at step two.
        headings = {s.value for s in at.subheader}
        assert "5 · Shear sonic" in headings
        assert "9 · Analysis window" in headings          # reached the end

    def test_a_missing_p_sonic_still_stops_it(self):
        """The relaxation is specific: there is no predicting a Vp or a density
        from what would be left."""
        well, raw, units, _ = self.stripped()
        well.df = well.df.drop(columns=["VP"])
        at = self.page(well, raw, units)
        assert not at.exception
        assert any("Missing required curve" in e.value for e in at.error)

    def test_it_offers_the_models_the_well_can_actually_support(self):
        well, raw, units, _ = self.stripped()
        at = self.page(well, raw, units)
        radio = next(r for r in at.radio if "Where Vs comes from" in r.label)
        assert "Greenberg-Castagna, mixed by VSH" in radio.options
        assert "Castagna mudrock line" in radio.options
        # No measured Vs anywhere, so there is no trend of its own to fit.
        assert "This well's own Vp-Vs trend" not in radio.options

    def test_the_prediction_lands_within_a_few_per_cent_of_the_real_log(self):
        well, raw, units, truth = self.stripped()
        at = self.page(well, raw, units)
        next(r for r in at.radio
             if "Where Vs comes from" in r.label).set_value(
                 "greenberg_castagna").run()
        next(b for b in at.button if "Apply shear sonic" in b.label).click().run()
        assert not at.exception

        got = at.session_state["well"].df["VS"].to_numpy(float)
        error = np.abs(got - truth) / truth
        assert np.nanmedian(error) < 0.06
        assert abs(np.nanmedian((got - truth) / truth)) < 0.02   # unbiased

    def test_what_it_predicted_is_recorded_per_sample(self):
        """Provenance, not a per-well flag: a partially logged well is the
        common case and 'this curve is 72% invented' is the useful statement."""
        well, raw, units, _ = self.stripped()
        at = self.page(well, raw, units)
        next(r for r in at.radio
             if "Where Vs comes from" in r.label).set_value(
                 "greenberg_castagna").run()
        next(b for b in at.button if "Apply shear sonic" in b.label).click().run()

        assert at.session_state["vs_source"] == "computed"
        mask = np.asarray(at.session_state["vs_predicted"], dtype=bool)
        # Predicted exactly where VSH allowed a prediction, nowhere else.
        vsh = np.isfinite(load()[0].df["VSH"].to_numpy(float))
        assert mask.sum() == int(vsh.sum())

    def test_the_page_says_the_classes_will_move(self):
        """The finding this warning exists for: 4.4% on the velocity is not
        4.4% on the answer."""
        well, raw, units, _ = self.stripped()
        at = self.page(well, raw, units)
        next(r for r in at.radio
             if "Where Vs comes from" in r.label).set_value(
                 "greenberg_castagna").run()
        next(b for b in at.button if "Apply shear sonic" in b.label).click().run()
        text = " ".join(w.value for w in at.warning)
        assert "different AVO class" in text

    def test_a_predicted_well_is_not_the_same_well(self):
        """Measured against predicted, straight through the pipeline. This is
        the number the warning quotes, and it must not drift away from it."""
        from avo_qi.analysis import reflector_analysis
        from avo_qi.core.vs_prediction import (fill_missing_vs,
                                               lithology_fractions_from_vsh,
                                               polynomial_vs)
        from avo_qi.ui import Settings

        measured = reflector_analysis(load()[0], Settings())["table"]

        well = load()[0]
        vp = well.df["VP"].to_numpy(float)
        vsh = well.df["VSH"].to_numpy(float)
        well.df["VS"] = fill_missing_vs(
            np.full(len(well.df), np.nan),
            polynomial_vs(vp, lithology_fractions_from_vsh(vsh)))["vs"]
        predicted = reflector_analysis(well, Settings())["table"]

        assert len(predicted) < len(measured)          # 18 against 25

        paired = agreed = 0
        for _, row in predicted.iterrows():
            near = measured[np.abs(measured["depth"] - row["depth"]) < 3.0]
            if len(near) == 1:
                paired += 1
                agreed += int(near.iloc[0]["avo_class"] == row["avo_class"])
        assert paired >= 8
        assert agreed / paired < 0.75      # ~58% agree; a long way from all


class TestAnisotropyOnARealWell:
    """What a shale fabric does to 15/9-19-A's classes.

    The demo well is a blocky synthetic whose VSH takes two values, so its
    anisotropy shifts are huge (0.11) and change nothing, because every
    reflector sits far from a class boundary. A real well is the useful test:
    gradational VSH, modest shifts, and labels that sit where they sit.

    The answer here is reassuring rather than dramatic, and is recorded as
    such: at a moderate literature shale nothing moves, and only the strongest
    one in Thomsen's range moves a single reflector. That is worth pinning
    precisely because it is the un-dramatic outcome — if a later change starts
    reclassifying this well on anisotropy, something is wrong.
    """

    @staticmethod
    def shifts(table, angles, shale):
        from avo_qi.core.anisotropy import (fitted_gradient_shift,
                                            thomsen_from_vsh)

        out = []
        for _, row in table.iterrows():
            upper = thomsen_from_vsh(row["vsh_above"], shale)
            lower = thomsen_from_vsh(row["vsh_below"], shale)
            out.append(np.nan if upper is None or lower is None
                       else fitted_gradient_shift(upper, lower, angles))
        return np.asarray(out, dtype=float)

    @staticmethod
    def analysis():
        from avo_qi.analysis import reflector_analysis
        from avo_qi.ui import Settings

        well, _, _ = load()
        return reflector_analysis(well, Settings())

    def test_five_reflectors_have_no_shale_volume_to_scale_with(self):
        """The 166 m with no VSH is why ``thomsen_from_vsh`` returns None
        rather than isotropy: those reflectors have an *unknown* fabric, and
        reporting "no shift" for them would be reporting a measurement nobody
        made."""
        from avo_qi.core.anisotropy import LITERATURE_SHALES

        found = self.analysis()
        shifts = self.shifts(found["table"], found["angles"],
                             LITERATURE_SHALES["moderate"])
        assert len(shifts) == 25
        assert int(np.isfinite(shifts).sum()) == 20

    def test_a_moderate_shale_moves_no_class_here(self):
        from avo_qi.core.anisotropy import LITERATURE_SHALES
        from avo_qi.core.avo import classify
        from avo_qi.ui import Settings

        found = self.analysis()
        table, angles = found["table"], found["angles"]
        shifts = self.shifts(table, angles, LITERATURE_SHALES["moderate"])
        a_tol = Settings().a_tol

        moved = [row["avo_class"] != classify(row["A_shuey"],
                                              row["B_shuey"] + shift, a_tol)
                 for (_, row), shift in zip(table.iterrows(), shifts)
                 if np.isfinite(shift)]
        assert sum(moved) == 0
        # ...and not because the shifts are nothing: they are real, just small
        # against where these reflectors sit.
        assert np.nanmax(np.abs(shifts)) == pytest.approx(0.0123, abs=5e-4)
        assert np.nanmax(np.abs(shifts)) < a_tol

    def test_the_strongest_literature_shale_moves_exactly_one(self):
        from avo_qi.core.anisotropy import LITERATURE_SHALES
        from avo_qi.core.avo import classify
        from avo_qi.ui import Settings

        found = self.analysis()
        table, angles = found["table"], found["angles"]
        shifts = self.shifts(table, angles, LITERATURE_SHALES["strong"])
        a_tol = Settings().a_tol

        moves = [(row["depth"], row["avo_class"],
                  classify(row["A_shuey"], row["B_shuey"] + shift, a_tol))
                 for (_, row), shift in zip(table.iterrows(), shifts)
                 if np.isfinite(shift)
                 and row["avo_class"] != classify(row["A_shuey"],
                                                  row["B_shuey"] + shift, a_tol)]
        assert len(moves) == 1
        depth, was, now = moves[0]
        assert (was, now) == ("IV", "III")
        assert depth == pytest.approx(3823, abs=2)

    def test_the_effect_grows_with_the_assumed_fabric(self):
        """Monotonic in the shale's delta, as it must be, and worth checking
        because nothing else here would notice a sign error in the scaling."""
        from avo_qi.core.anisotropy import LITERATURE_SHALES

        found = self.analysis()
        biggest = [np.nanmax(np.abs(self.shifts(
            found["table"], found["angles"], LITERATURE_SHALES[name])))
            for name in ("weak", "moderate", "strong")]
        assert biggest == sorted(biggest)
        assert biggest[0] == pytest.approx(0.0036, abs=3e-4)

    def test_a_reflector_with_no_shale_contrast_gets_no_shift(self):
        """The contrast physics, seen on real data: where the two lobes carry
        the same shale volume the fabric cancels exactly, however strong it
        is. If this ever became non-zero the model would have turned
        anisotropy into a property of a rock instead of of an interface."""
        from avo_qi.core.anisotropy import LITERATURE_SHALES

        found = self.analysis()
        table, angles = found["table"], found["angles"]
        shifts = self.shifts(table, angles, LITERATURE_SHALES["strong"])
        contrast = (table["vsh_below"] - table["vsh_above"]).to_numpy(float)

        flat = np.isfinite(shifts) & (np.abs(contrast) < 1e-9)
        if flat.any():
            assert np.allclose(shifts[flat], 0.0, atol=1e-15)
        # And the shift tracks the contrast where there is one.
        both = np.isfinite(shifts) & np.isfinite(contrast)
        assert np.corrcoef(shifts[both], contrast[both])[0, 1] > 0.99
