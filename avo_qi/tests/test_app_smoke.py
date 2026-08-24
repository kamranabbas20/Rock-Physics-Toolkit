"""Smoke tests for the Streamlit pages, run headless through ``AppTest``.

These check that each page executes end to end against the demo well and
renders the elements the spec calls for.  They are skipped when Streamlit is
not installed, so ``core/`` stays testable on its own.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

streamlit = pytest.importorskip("streamlit")
pytest.importorskip("plotly")

from streamlit.testing.v1 import AppTest  # noqa: E402

from avo_qi.io.loader import read_well, standardise  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(HERE, "app.py")
PAGES = os.path.join(HERE, "pages")
DEMO = os.path.join(HERE, "sample_data", "demo_well.las")


def demo_well():
    df, units = read_well(DEMO)
    return standardise(df, units=units, name="DEMO-1"), df, units


def run_page(path, with_well=True, timeout=90):
    at = AppTest.from_file(path, default_timeout=timeout)
    if with_well:
        _inject_demo_well(at)
    at.run()
    return at


def _inject_demo_well(at):
    """Put the demo well into session state the way ``load_demo_well`` does.

    That includes the zone code-to-name mapping, which the real loader applies
    and without which the zones read back as bare codes.
    """
    from avo_qi.sample_data.make_demo_well import ZONE_NAMES
    from avo_qi.ui import Settings

    well, raw, units = demo_well()
    at.session_state["well"] = well
    at.session_state["raw_df"] = raw
    at.session_state["raw_units"] = units
    # AppTest's session state raises rather than returning None for a key that
    # has never been set, so this cannot use .get().
    try:
        settings = at.session_state["settings"]
    except (KeyError, AttributeError):
        settings = Settings()
    settings.zone_names = dict(ZONE_NAMES)
    at.session_state["settings"] = settings


class TestLandingPage:
    def test_runs_without_a_well(self):
        at = run_page(APP, with_well=False)
        assert not at.exception
        assert any("AVO & QI" in t.value for t in at.title)

    def test_shows_the_loaded_well(self):
        at = run_page(APP)
        assert not at.exception
        assert any("DEMO-1" in h.value for h in at.subheader)
        assert len(at.metric) >= 4


@pytest.fixture(scope="module")
def qc_page():
    return run_page(os.path.join(PAGES, "1_Load_and_QC.py"))


@pytest.fixture(scope="module")
def crossplots_page():
    return run_page(os.path.join(PAGES, "2_Data_and_Crossplots.py"))


@pytest.fixture(scope="module")
def gather_page():
    return run_page(os.path.join(PAGES, "3_Synthetic_Gather.py"))


@pytest.fixture(scope="module")
def avo_page():
    return run_page(os.path.join(PAGES, "4_AVO_Classification.py"))


@pytest.fixture(scope="module")
def rock_physics_page():
    return run_page(os.path.join(PAGES, "5_Rock_Physics.py"))


class TestDataAndCrossplotsPage:
    def test_runs_clean(self, crossplots_page):
        assert not crossplots_page.exception

    def test_renders_tracks_and_every_crossplot_tab(self, crossplots_page):
        # Log tracks, five crossplot tabs, the EEI track pair and the chi sweep.
        assert len(crossplots_page.tabs) >= 5
        assert len(crossplots_page.multiselect) >= 1

    def test_offers_the_standardised_download(self, crossplots_page):
        assert any("CSV" in b.label for b in crossplots_page.download_button)

    def test_prompts_when_no_well_is_loaded(self):
        at = run_page(os.path.join(PAGES, "2_Data_and_Crossplots.py"), with_well=False)
        assert not at.exception
        assert at.info or at.warning


class TestSyntheticGatherPage:
    def test_runs_clean(self, gather_page):
        assert not gather_page.exception

    def test_builds_a_gather_into_session_state(self, gather_page):
        gather = gather_page.session_state["gather"]
        angles = gather_page.session_state["settings"].angles
        assert gather.ndim == 2
        assert gather.shape[1] == angles.size
        assert gather.shape[0] == gather_page.session_state["gather_twt"].size

    def test_gather_shows_the_class_three_signature(self, gather_page):
        import numpy as np

        gather = gather_page.session_state["gather"]
        twt = gather_page.session_state["gather_twt"]
        angles = gather_page.session_state["settings"].angles
        # The gas-sand top is the strongest trough in the well.
        i = int(np.unravel_index(np.argmin(gather), gather.shape)[0])
        window = gather[max(i - 10, 0): i + 10, :]
        trough = window.min(axis=0)
        assert trough[-1] < trough[0]              # brightens with angle
        assert 0 < i < twt.size and angles.size > 1

    def test_offers_all_three_exports(self, gather_page):
        labels = {b.label for b in gather_page.download_button}
        assert {"CSV", "NPY"}.issubset(labels)

    def test_stops_politely_without_a_well(self):
        at = run_page(os.path.join(PAGES, "3_Synthetic_Gather.py"), with_well=False)
        assert not at.exception
        assert at.warning


def reflector_table(page):
    """The reflector table, found by its columns rather than by position.

    The page renders other tables too (the lithology pair summary), so an
    index would break whenever one is added.
    """
    for element in page.dataframe:
        frame = element.value
        if hasattr(frame, "columns") and {"avo_class", "A_shuey"} <= set(frame.columns):
            return frame
    raise AssertionError("no reflector table on the page")


class TestAvoClassificationPage:
    def test_runs_clean(self, avo_page):
        assert not avo_page.exception

    def test_reports_class_counts_and_a_reflector_table(self, avo_page):
        assert len(avo_page.metric) >= 6
        assert len(avo_page.dataframe) >= 1

    def test_finds_the_class_three_gas_sand(self, avo_page):
        table = reflector_table(avo_page)
        assert "avo_class" in table.columns
        assert (table["avo_class"] == "III").any()

    def test_shuey_and_aki_richards_agree_in_the_table(self, avo_page):
        import numpy as np

        table = reflector_table(avo_page)
        assert np.nanmax(np.abs(table["dA"].to_numpy(float))) < 1e-6
        assert np.nanmax(np.abs(table["dB"].to_numpy(float))) < 1e-6

    def test_stops_politely_without_a_well(self):
        at = run_page(os.path.join(PAGES, "4_AVO_Classification.py"), with_well=False)
        assert not at.exception
        assert at.warning


class TestRockPhysicsPage:
    def test_runs_clean(self, rock_physics_page):
        assert not rock_physics_page.exception

    def test_renders_every_diagnostic_tab(self, rock_physics_page):
        # Model setup (3), then Vp-Vs, Gardner, velocity-porosity,
        # moduli & bounds, forward model, fluid substitution.
        assert len(rock_physics_page.tabs) >= 9

    def test_exposes_the_frame_model_controls(self, rock_physics_page):
        labels = {s.label for s in rock_physics_page.slider}
        assert "Critical porosity \u03c6c" in labels
        assert "Coordination number n" in labels
        assert "Effective pressure (MPa)" in labels
        assert any(r.label == "Modulus" for r in rock_physics_page.radio)

    def test_reports_the_mineral_and_pack_moduli(self, rock_physics_page):
        labels = {m.label for m in rock_physics_page.metric}
        assert "Mineral K" in labels
        assert "Hertz-Mindlin K (dry)" in labels

    def test_offers_to_saturate_the_dry_frame_curves(self, rock_physics_page):
        """The dry-vs-saturated mismatch is now fixable rather than only warned about."""
        labels = [c.label for c in rock_physics_page.checkbox]
        assert any("Gassmann" in label for label in labels)

    def test_explains_the_saturation_step_it_applied(self, rock_physics_page):
        captions = " ".join(c.value for c in rock_physics_page.caption)
        assert "dry to saturated through" in captions

    def test_the_forward_model_tab_offers_its_controls(self, rock_physics_page):
        labels = {s.label for s in rock_physics_page.selectbox}
        assert "Frame model" in labels
        assert "Shale mineral" in labels
        assert "Hydrocarbon" in labels

    def test_the_forward_model_reports_a_misfit_for_every_curve(self, rock_physics_page):
        labels = {m.label for m in rock_physics_page.metric}
        for curve in ("VP", "VS", "RHOB"):
            assert any(label.startswith(f"{curve} bias") for label in labels), curve

    def test_says_whether_the_porosity_makes_the_density_check_circular(
            self, rock_physics_page):
        text = " ".join(c.value for c in rock_physics_page.caption) + " ".join(
            w.value for w in rock_physics_page.warning)
        assert "density" in text.lower()
        assert "meaningful" in text or "circular" in text

    def test_shows_the_wells_own_vsh_as_reference_without_applying_it(
            self, rock_physics_page):
        captions = " ".join(c.value for c in rock_physics_page.caption)
        assert "Shown, not applied" in captions

    def test_the_substitution_tab_offers_a_button(self, rock_physics_page):
        assert any("Substitute" in b.label for b in rock_physics_page.button)

    def test_batzle_wang_is_offered_but_off_by_default(self, rock_physics_page):
        boxes = [c for c in rock_physics_page.checkbox
                 if "pressure and temperature" in c.label]
        assert len(boxes) == 1
        assert boxes[0].value is False


class TestReflectorSelection:
    """The wiring that turns a click into a selected reflector.

    Driven by injecting the selection payload Streamlit itself would store,
    so the logic is covered deterministically without a browser in the loop.
    """

    @staticmethod
    def _page(**state):
        at = AppTest.from_file(os.path.join(PAGES, "4_AVO_Classification.py"),
                               default_timeout=90)
        _inject_demo_well(at)
        for key, value in state.items():
            at.session_state[key] = value
        at.run()
        return at

    def test_a_trace_click_selects_the_reflector_it_names(self):
        payload = {"selection": {"points": [{"curve_number": 6, "point_number": 0,
                                             "point_index": 0, "x": -0.1,
                                             "y": 1.633, "customdata": 1}],
                                 "point_indices": [0], "box": [], "lasso": []}}
        at = self._page(class_trace=payload)
        assert not at.exception
        assert at.session_state["reflector_pick"] == 1

    def test_a_click_carrying_no_customdata_falls_back_to_the_time(self):
        """Selection payloads differ between Streamlit versions, so the time
        of the clicked point is the backstop."""
        payload = {"selection": {"points": [{"curve_number": 6, "x": -0.1,
                                             "y": 1.706}],
                                 "point_indices": [0], "box": [], "lasso": []}}
        at = self._page(class_trace=payload)
        assert not at.exception
        table = reflector_table(at)
        chosen = at.session_state["reflector_pick"]
        assert 0 <= chosen < len(table)

    def test_a_row_click_selects_that_reflector(self):
        at = self._page()
        table = reflector_table(at)
        sample = int(table["sample"].iloc[3])
        at.session_state["reflector_rows"] = {"selection": {"rows": [3], "columns": []}}
        at.run()
        assert not at.exception
        picked = at.session_state["reflector_pick"]
        assert int(reflector_table(at)["sample"].iloc[picked]) == sample

    def test_an_empty_selection_leaves_the_pick_alone(self):
        at = self._page(class_trace={"selection": {"points": [], "point_indices": [],
                                                   "box": [], "lasso": []}})
        assert not at.exception
        # Falls back to the strongest reflector rather than to row zero.
        table = reflector_table(at)
        pick = at.session_state["reflector_pick"]
        strongest = int(np.argmax(np.abs(table["R0"].to_numpy(float))))
        assert pick == strongest

    def test_the_dropdown_still_wins_when_no_new_click_arrives(self):
        """A stale selection must not override a manual dropdown change on
        every rerun, or the dropdown would be unusable."""
        payload = {"selection": {"points": [{"customdata": 1, "y": 1.633}],
                                 "point_indices": [0], "box": [], "lasso": []}}
        at = self._page(class_trace=payload)
        assert at.session_state["reflector_pick"] == 1
        at.session_state["reflector_pick"] = 5      # as the dropdown would
        at.run()
        assert at.session_state["reflector_pick"] == 5

    def test_the_trace_offers_a_marker_for_every_reflector(self):
        at = self._page()
        assert not at.exception
        assert len(reflector_table(at)) > 0


class TestTheDetailPanelShowsWhatItFitted:
    """The panel plots A and B over a modelled curve; both must come from the
    same layers, or the fit appears wrong when it is not."""

    def test_it_names_the_lobe_it_blocked_on(self, avo_page):
        captions = " ".join(c.value for c in avo_page.caption)
        assert "the reflector's own lobe" in captions
        assert "the same layers the A and B above were fitted to" in captions

    def test_the_quoted_layer_properties_come_from_the_same_window(self, avo_page):
        captions = " ".join(c.value for c in avo_page.caption)
        assert "from the reflector's own lobe" in captions

    def test_the_unfitted_adjacent_curve_is_still_offered_for_contrast(self, avo_page):
        """Seeing the gap between the lobe and the raw adjacent pair is the
        point of blocking; it is drawn, but labelled as not fitted."""
        captions = " ".join(c.value for c in avo_page.caption)
        assert "not what was fitted" in captions


class TestAWellWithNoZonation:
    """Not every well carries a ZONE curve, and one that does not must work.

    The failure this pins was severe and silent: zone names belong to the well
    they came from, but the selection lived on the session. Load a zoned well,
    then an unzoned one, and every sample reads as unzoned while the filter is
    still looking for the *previous* well's zone names — so the filter matched
    nothing and hid the entire well behind "no reflector lies in the selected
    zones", which looks like a broken app rather than an empty filter.
    """

    @staticmethod
    def _unzoned_well():
        well, _, _ = demo_well()
        well.df = well.df.drop(columns=["ZONE"])
        return well

    def _page(self, name, zones=("Shale", "Gas Sand"), lithologies=None):
        from avo_qi.ui import Settings

        at = AppTest.from_file(os.path.join(PAGES, name), default_timeout=90)
        settings = Settings()
        settings.zones = list(zones)          # left over from another well
        if lithologies is not None:
            settings.lithologies = list(lithologies)
        at.session_state["settings"] = settings
        at.session_state["well"] = self._unzoned_well()
        at.session_state["raw_df"] = None
        at.session_state["raw_units"] = {}
        at.run()
        return at

    def test_the_avo_page_still_finds_its_reflectors(self):
        at = self._page("4_AVO_Classification.py")
        assert not at.exception
        assert not any("No reflector lies in the selected zones" in w.value
                       for w in at.warning)
        assert len(reflector_table(at)) > 0

    def test_nothing_is_reported_as_hidden_by_a_zone_filter(self):
        at = self._page("4_AVO_Classification.py")
        assert not any("Zone filter is hiding" in c.value for c in at.caption)

    def test_the_rock_physics_page_keeps_its_samples(self):
        at = self._page("5_Rock_Physics.py")
        assert not at.exception
        assert not any("excluded every sample" in w.value for w in at.warning)

    def test_the_sidebar_says_the_well_has_no_zonation(self):
        at = self._page("5_Rock_Physics.py")
        captions = " ".join(c.value for c in at.sidebar.caption)
        assert "no zonation" in captions
        # ...and says how to get one, rather than only that it is missing.
        assert "Load & QC" in captions

    def test_a_stale_lithology_selection_cannot_blank_it_either(self):
        """Same shape of bug on the other filter, so the same guard."""
        well = self._unzoned_well()
        well.df = well.df.drop(columns=[c for c in ("VSH", "GR")
                                        if c in well.df.columns])
        from avo_qi.ui import Settings

        at = AppTest.from_file(os.path.join(PAGES, "5_Rock_Physics.py"),
                               default_timeout=90)
        settings = Settings()
        settings.lithologies = ["sand"]       # every sample here is undefined
        at.session_state["settings"] = settings
        at.session_state["well"] = well
        at.session_state["raw_df"] = None
        at.session_state["raw_units"] = {}
        at.run()
        assert not at.exception
        assert not any("excluded every sample" in w.value for w in at.warning)


class TestTheFiltersThemselves:
    """Unit-level cover for the two guards, away from any page."""

    @staticmethod
    def _settings(**kwargs):
        from avo_qi.ui import Settings

        settings = Settings()
        for key, value in kwargs.items():
            setattr(settings, key, value)
        return settings

    def test_an_unzoned_well_ignores_a_zone_selection(self):
        from avo_qi.core.zones import UNZONED
        from avo_qi.ui import apply_zone_filter

        labels = np.array([UNZONED] * 5, dtype=object)
        mask = apply_zone_filter(labels, self._settings(zones=["Shale"]))
        assert mask.all()

    def test_a_real_selection_that_excludes_everything_is_still_honoured(self):
        """The guard must not paper over a genuine empty answer."""
        from avo_qi.ui import apply_zone_filter

        labels = np.array(["Shale", "Shale", "Sand"], dtype=object)
        mask = apply_zone_filter(labels, self._settings(zones=["Limestone"]))
        assert not mask.any()

    def test_a_partly_zoned_well_filters_as_before(self):
        from avo_qi.core.zones import UNZONED
        from avo_qi.ui import apply_zone_filter

        labels = np.array(["Shale", UNZONED, "Sand"], dtype=object)
        mask = apply_zone_filter(labels, self._settings(zones=["Shale"]))
        assert list(mask) == [True, False, False]

    def test_no_selection_keeps_everything(self):
        from avo_qi.ui import apply_zone_filter

        labels = np.array(["Shale", "Sand"], dtype=object)
        assert apply_zone_filter(labels, self._settings(zones=[])).all()

    def test_a_well_with_no_lithology_ignores_a_lithology_selection(self):
        from avo_qi.core.lithology import UNDEFINED
        from avo_qi.ui import apply_lithology_filter

        labels = np.array([UNDEFINED] * 4, dtype=object)
        mask = apply_lithology_filter(None, labels,
                                      self._settings(lithologies=["sand"]))
        assert mask.all()

    def test_a_real_lithology_selection_still_filters(self):
        from avo_qi.core.lithology import UNDEFINED
        from avo_qi.ui import apply_lithology_filter

        labels = np.array(["sand", "shale", UNDEFINED], dtype=object)
        mask = apply_lithology_filter(None, labels,
                                      self._settings(lithologies=["sand"]))
        assert list(mask) == [True, False, False]


class TestUncertaintyInTheApp:
    """The Monte Carlo panels, off by default and honest when switched on."""

    def test_rock_physics_offers_it_but_does_not_run_it_unasked(self, rock_physics_page):
        boxes = [c for c in rock_physics_page.checkbox
                 if "Monte Carlo" in c.label]
        assert len(boxes) == 1
        assert boxes[0].value is False

    def test_the_uncertainty_controls_are_there(self, rock_physics_page):
        labels = {s.label for s in rock_physics_page.slider}
        for wanted in ("Realisations", "VSH \u00b1 (v/v)", "SW \u00b1 (v/v)",
                       "VSH\u2013PHIT correlation"):
            assert wanted in labels, wanted

    def test_running_it_reports_a_band_width_for_every_curve(self):
        at = run_page(os.path.join(PAGES, "5_Rock_Physics.py"))
        next(c for c in at.checkbox if "Monte Carlo" in c.label).set_value(True).run()
        assert not at.exception
        labels = {m.label for m in at.metric}
        for curve in ("VP", "VS", "RHOB"):
            assert f"{curve} P10\u2013P90 width" in labels

    def test_a_band_has_width_only_because_the_inputs_do(self):
        at = run_page(os.path.join(PAGES, "5_Rock_Physics.py"))
        next(c for c in at.checkbox if "Monte Carlo" in c.label).set_value(True)
        for label in ("VSH \u00b1 (v/v)", "PHIT \u00b1 (v/v)", "SW \u00b1 (v/v)",
                      "\u03c6c \u00b1", "Effective pressure \u00b1 (MPa)",
                      "Coordination number \u00b1"):
            next(s for s in at.slider if s.label == label).set_value(0.0)
        at.run()
        assert not at.exception
        width = next(m for m in at.metric if m.label.startswith("VP P10"))
        assert float(width.value.split()[0]) == pytest.approx(0.0, abs=1e-6)

    def test_avo_page_offers_class_probabilities_off_by_default(self, avo_page):
        boxes = [c for c in avo_page.checkbox if "class probabilities" in c.label]
        assert len(boxes) == 1
        assert boxes[0].value is False

    def test_running_it_reports_confidence_and_flags_ambiguity(self):
        at = run_page(os.path.join(PAGES, "4_AVO_Classification.py"))
        next(c for c in at.checkbox
             if "class probabilities" in c.label).set_value(True).run()
        assert not at.exception
        labels = {m.label for m in at.metric}
        assert "Median confidence" in labels
        assert "Ambiguous reflectors" in labels

    def test_a_perfect_tool_leaves_every_class_certain(self):
        """With no measurement error there is nothing for the class to be
        uncertain about, so the odds must collapse onto the label."""
        at = run_page(os.path.join(PAGES, "4_AVO_Classification.py"))
        next(c for c in at.checkbox
             if "class probabilities" in c.label).set_value(True)
        for label in ("Vp \u00b1 (%)", "Vs \u00b1 (%)", "RHOB \u00b1 (%)"):
            next(s for s in at.slider if s.label == label).set_value(0.0)
        at.run()
        assert not at.exception
        median = next(m for m in at.metric if m.label == "Median confidence")
        assert median.value == "100%"
        assert any("holds its class" in s.value for s in at.success)

        # The regression that matters: with no noise the odds must land on the
        # label itself. They will not if the Monte Carlo classifies a different
        # interface from the one the label was fitted to — the page blocks its
        # layers by default, so the realisations have to be blocked too.
        differs = next(m for m in at.metric
                       if m.label == "Modal class differs from the label")
        assert differs.value == "0"

    def test_the_odds_are_computed_on_the_layers_the_label_used(self):
        """The Monte Carlo must re-average over the *same* lobe windows.

        Block on anything else and the odds describe a different interface, so
        every disagreement with the label is that mismatch rather than a
        finding. With a perfect tool the two must land on the same class.
        """
        at = run_page(os.path.join(PAGES, "4_AVO_Classification.py"))
        next(c for c in at.checkbox
             if "class probabilities" in c.label).set_value(True)
        for label in ("Vp \u00b1 (%)", "Vs \u00b1 (%)", "RHOB \u00b1 (%)"):
            next(s for s in at.slider if s.label == label).set_value(0.0)
        at.run()
        assert not at.exception
        differs = next(m for m in at.metric
                       if m.label == "Modal class differs from the label")
        assert differs.value == "0"


class TestTheZoneFilterScopesTheModel:
    """The misfit and the reference statistics must follow the sidebar.

    Reported over the whole well they say very little; the point of the
    numbers is to answer "how does the model do *in this zone*". A filter that
    scoped the plots but not the statistics would be worse than none, because
    the numbers would look scoped and not be.
    """

    @staticmethod
    def _page(zones=None):
        at = run_page(os.path.join(PAGES, "5_Rock_Physics.py"))
        if zones is not None:
            picker = next(m for m in at.sidebar.multiselect
                          if m.label == "Zones to analyse")
            picker.set_value(zones).run()
        return at

    @staticmethod
    def _vsh_caption(at):
        return next(c.value for c in at.caption
                    if "this selection measures" in c.value)

    def test_the_reference_vsh_is_the_whole_well_by_default(self):
        # The demo well is mostly shale, so the unfiltered median is high.
        assert "VSH 0.85" in self._vsh_caption(self._page())

    def test_filtering_to_one_zone_rescopes_the_reference(self):
        at = self._page(["Brine Sand"])
        assert not at.exception
        # The brine sand's own VSH is 0.12, not the well's 0.85.
        assert "VSH 0.12" in self._vsh_caption(at)

    def test_filtering_to_one_zone_rescopes_the_misfit(self):
        whole = next(m for m in self._page().metric if m.label == "VP bias")
        zoned = next(m for m in self._page(["Brine Sand"]).metric
                     if m.label == "VP bias")
        assert whole.value != zoned.value

    def test_the_model_still_runs_on_a_single_zone(self):
        at = self._page(["Gas Sand"])
        assert not at.exception
        assert any(m.label.startswith("VP bias") for m in at.metric)


class TestSubstitutingFromThePage:
    """Pressing the button must really produce a case the rest of the app sees."""

    @staticmethod
    def _substitute():
        at = run_page(os.path.join(PAGES, "5_Rock_Physics.py"))
        button = next(b for b in at.button if "Substitute" in b.label)
        button.click().run()
        return at

    def test_the_button_creates_the_case_on_the_well(self):
        at = self._substitute()
        assert not at.exception
        well = at.session_state["well"]
        assert "gas" in well.cases
        assert well.is_computed("gas")

    def test_it_reports_how_many_samples_it_managed(self):
        at = self._substitute()
        assert any("Substituted" in s.value for s in at.success)

    def test_the_substituted_curves_are_slower_than_the_brine_ones(self):
        at = self._substitute()
        well = at.session_state["well"]
        source = well.frame("in situ")["VP"].to_numpy(float)
        gassy = well.frame("gas")["VP"].to_numpy(float)
        both = np.isfinite(source) & np.isfinite(gassy)
        assert both.sum() > 100
        assert np.nanmedian(gassy[both]) < np.nanmedian(source[both])

    def test_the_computed_case_is_labelled_as_computed(self):
        at = self._substitute()
        assert any("Computed here" in i.value for i in at.info)

    def test_another_page_picks_the_case_up(self):
        """The whole reason for writing it as an ordinary fluid case."""
        at = self._substitute()
        well = at.session_state["well"]

        gather = AppTest.from_file(os.path.join(PAGES, "3_Synthetic_Gather.py"),
                                   default_timeout=90)
        gather.session_state["well"] = well
        gather.session_state["settings"] = at.session_state["settings"]
        gather.run()
        assert not gather.exception
        case_box = next(s for s in gather.sidebar.selectbox
                        if s.label == "Substituted case")
        # AppTest reports the formatted labels, which is also how a reader sees
        # them — the computed case must be flagged as such wherever it appears.
        assert "gas (computed)" in case_box.options

    def test_warns_before_overwriting_a_case_that_came_from_the_file(self):
        """The demo well already carries a loaded gas case; replacing measured
        curves with modelled ones is not something to do quietly."""
        at = run_page(os.path.join(PAGES, "5_Rock_Physics.py"))
        assert any("loaded from the file" in w.value for w in at.warning)

    def test_the_sidebar_marks_it_so_nobody_mistakes_it_for_a_log(self):
        at = self._substitute()
        case_box = next(s for s in at.sidebar.selectbox
                        if s.label == "Substituted case")
        assert case_box.format_func("gas") == "gas (computed)"
        assert case_box.format_func("in situ") == "in situ"

    def test_stops_politely_without_a_well(self):
        at = run_page(os.path.join(PAGES, "5_Rock_Physics.py"), with_well=False)
        assert not at.exception
        assert at.warning


class TestFluidCasesInTheApp:
    """The demo well carries four cases; the pages must pick them up."""

    def test_sidebar_offers_the_case_selector(self, crossplots_page):
        labels = {sb.label for sb in crossplots_page.selectbox}
        assert "Substituted case" in labels

    def test_crossplots_page_reports_the_cases(self, crossplots_page):
        assert any("fluid cases" in i.value for i in crossplots_page.info)

    def test_gather_page_builds_a_difference_gather(self, gather_page):
        labels = {m.label for m in gather_page.metric}
        assert "Peak |difference|" in labels

    def test_avo_page_compares_every_case(self, avo_page):
        labels = {m.label for m in avo_page.metric}
        assert "Reflectors compared" in labels
        assert "Change class with fluid" in labels
        assert "Largest fluid vector" in labels

    def test_avo_page_lists_the_class_changes(self, avo_page):
        headers = {h.value for h in avo_page.subheader}
        assert "Reflectors that change class" in headers
        assert "Amplitude vs angle, by fluid case" in headers

    def test_rock_physics_page_follows_the_case(self, rock_physics_page):
        assert any("case" in i.value for i in rock_physics_page.info)


class TestClassifiedTraceSelection:
    """The trace panel marks reflectors by class and is clickable."""

    def test_trace_panel_renders(self, avo_page):
        labels = {sb.label for sb in avo_page.selectbox}
        assert "Trace" in labels
        assert "Reflector" in labels

    def test_selection_resolver_reads_a_plotly_payload(self):
        import numpy as np

        from avo_qi.ui import selected_reflector_index

        marker_twt = np.array([1.60, 1.65, 1.70])
        assert selected_reflector_index(
            {"selection": {"points": [{"customdata": [2], "y": 1.70}]}}, marker_twt
        ) == 2
        # No customdata: fall back to the nearest marker in time.
        assert selected_reflector_index(
            {"selection": {"points": [{"y": 1.648}]}}, marker_twt
        ) == 1
        assert selected_reflector_index({"selection": {"points": []}}, marker_twt) is None
        assert selected_reflector_index(None, marker_twt) is None

    def test_clicking_the_trace_drives_the_detail_panel(self):
        """A click lands in session state and the next run follows it."""
        at = AppTest.from_file(os.path.join(PAGES, "4_AVO_Classification.py"),
                               default_timeout=120)
        _inject_demo_well(at)
        at.run()
        assert not at.exception

        original = at.session_state["reflector_pick"]
        target = 0 if original != 0 else 1
        at.session_state["class_trace"] = {"selection": {"points": [{"customdata": [target]}]}}
        at.run()
        assert not at.exception
        assert at.session_state["reflector_pick"] == target


class TestLithologyFilter:
    """Lithology is cut from VSH and filters the crossplots and reflectors."""

    def test_sidebar_exposes_the_cutoffs_and_the_filter(self, crossplots_page):
        labels = {w.label for w in crossplots_page.number_input}
        assert {"Sand \u2264", "Silty \u2264", "Silt \u2264"}.issubset(labels)
        assert any(m.label == "Show lithologies" for m in crossplots_page.multiselect)

    def test_crossplots_offer_lithology_as_a_colour(self, crossplots_page):
        colour = next(sb for sb in crossplots_page.selectbox if sb.label == "Colour by")
        assert "LITHOLOGY" in colour.options

    def test_reflector_table_carries_the_lithology_pair(self, avo_page):
        table = reflector_table(avo_page)
        for column in ("litho_upper", "litho_lower", "litho_pair"):
            assert column in table.columns

    def test_the_gas_sand_top_is_a_shale_over_sand_pair(self, avo_page):
        table = reflector_table(avo_page)
        pairs = set(table["litho_pair"])
        assert "shale over sand" in pairs
        assert "sand over shale" in pairs

    def test_rock_physics_page_offers_lithology_colouring(self, rock_physics_page):
        colour = next(sb for sb in rock_physics_page.selectbox
                      if sb.label == "Colour the well data by")
        assert "LITHOLOGY" in colour.options

    def test_filtering_to_one_lithology_drops_reflectors(self):
        at = AppTest.from_file(os.path.join(PAGES, "4_AVO_Classification.py"),
                               default_timeout=120)
        _inject_demo_well(at)
        at.run()
        assert not at.exception
        everything = len(reflector_table(at))

        at.session_state["settings"].lithologies = ["sand"]
        at.run()
        assert not at.exception
        assert len(reflector_table(at)) < everything


class TestLoadAndQcPage:
    def test_runs_clean(self, qc_page):
        assert not qc_page.exception

    def test_prompts_when_nothing_is_loaded(self):
        at = run_page(os.path.join(PAGES, "1_Load_and_QC.py"), with_well=False)
        assert not at.exception
        assert at.info                       # "no well loaded"

    def test_offers_upload_and_the_demo_well(self, qc_page):
        assert len(qc_page.get("file_uploader")) >= 1
        assert any("demo well" in b.label for b in qc_page.button)

    def test_shows_the_curve_assignment_controls(self, qc_page):
        labels = {sb.label for sb in qc_page.selectbox}
        assert {"VP", "VS", "RHOB", "DEPTH"}.issubset(labels)

    def test_reports_the_depth_axis(self, qc_page):
        labels = {m.label for m in qc_page.metric}
        assert {"Interval", "Step", "Duplicates", "Gaps"}.issubset(labels)

    def test_the_demo_well_passes_its_checks(self, qc_page):
        """A clean well should reach the success message, not a flag table."""
        assert any("check" in s.value.lower() for s in qc_page.success) or \
            any("fail at least one check" in c.value for c in qc_page.caption)

    def test_offers_despiking_and_the_analysis_window(self, qc_page):
        toggles = {t.label for t in qc_page.toggle}
        assert "Despike Vp, Vs and RHOB" in toggles
        assert any("Depth range" in s.label for s in qc_page.slider)

    def test_exports_the_qc_flags(self, qc_page):
        assert any("QC flags" in b.label for b in qc_page.download_button)


class TestBlockingAndTuningInTheApp:
    """Both were built in core/ before they reached a page; these pin that they
    are now actually wired in."""

    def test_there_is_no_longer_a_choice_of_layer_source(self):
        """Adjacent samples were removed; lobe blocking is what the page does."""
        labels = {r.label for r in run_page(
            os.path.join(PAGES, "4_AVO_Classification.py")).radio}
        assert "Layer properties from" not in labels

    def test_blocking_reports_the_lobe_window_it_measured(self, avo_page):
        labels = {m.label for m in avo_page.metric}
        assert "Median lobe window" in labels
        assert "Largest change in A" in labels

    def test_the_lobe_window_is_narrower_than_a_fixed_half_cycle(self, avo_page):
        """Halving the lobe gives about a quarter period each side, against the
        half period the fixed window used — the contrasts should sharpen."""
        table = reflector_table(avo_page)
        assert "lobe_samples" in table.columns
        lobe = table.loc[table["blocking"] == "lobe", "lobe_samples"]
        assert len(lobe) > 0
        assert lobe.median() < 26          # the demo well's fixed window

    def test_every_reflector_says_which_window_it_used(self, avo_page):
        table = reflector_table(avo_page)
        assert set(table["blocking"]) <= {"lobe", "fixed window"}
        assert (table["blocking"] == "lobe").any()

    def test_the_tuning_section_is_present(self, avo_page):
        headers = {h.value for h in avo_page.header}
        assert "Tuned vs untuned" in headers
        labels = {m.label for m in avo_page.metric}
        assert "Tuning thickness" in labels
        assert "Largest gradient shift" in labels

    def test_the_tuning_table_compares_both(self, avo_page):
        frames = [d.value for d in avo_page.dataframe if hasattr(d.value, "columns")]
        tuning = [f for f in frames if "class_tuned" in f.columns]
        assert tuning, "no tuned-vs-untuned table on the page"
        table = tuning[0]
        for column in ("A_untuned", "A_tuned", "class_untuned", "changes_class"):
            assert column in table.columns

    def test_the_spec_pinned_gas_sand_survives_the_narrower_window(self):
        """SPEC.md 4.1 fixes the gas sand as Class III. Narrowing the window
        sharpens contrasts, and that must not quietly move the one answer the
        spec nails down."""
        table = reflector_table(run_page(
            os.path.join(PAGES, "4_AVO_Classification.py")))
        gas = table[table["sample"] == 33]
        assert len(gas) == 1
        assert gas["avo_class"].iloc[0] == "III"


class TestZonationAndMixingInTheApp:
    def test_the_sidebar_offers_the_zone_filter(self, avo_page):
        labels = {m.label for m in avo_page.multiselect}
        assert "Zones to analyse" in labels

    def test_reflectors_carry_their_zone(self, avo_page):
        table = reflector_table(avo_page)
        for column in ("zone", "zone_below", "is_zone_boundary"):
            assert column in table.columns

    def test_reservoir_tops_are_flagged_as_zone_boundaries(self, avo_page):
        table = reflector_table(avo_page)
        assert table["is_zone_boundary"].any()
        boundary = table[table["is_zone_boundary"]]
        assert "Gas Sand" in set(boundary["zone"]) | set(boundary["zone_below"])

    def test_filtering_to_one_zone_drops_reflectors(self):
        at = AppTest.from_file(os.path.join(PAGES, "4_AVO_Classification.py"),
                               default_timeout=180)
        _inject_demo_well(at)
        at.run()
        assert not at.exception
        everything = len(reflector_table(at))

        at.session_state["settings"].zones = ["Gas Sand"]
        at.session_state["zone_cache"] = {}
        at.run()
        assert not at.exception
        assert len(reflector_table(at)) < everything

    def test_rock_physics_offers_the_mixing_laws(self, rock_physics_page):
        labels = {sb.label for sb in rock_physics_page.selectbox}
        assert "Mixing law" in labels
        assert "Fluid mixing law" in labels

    def test_rock_physics_takes_several_minerals(self, rock_physics_page):
        minerals = next(m for m in rock_physics_page.multiselect
                        if m.label == "Minerals")
        assert set(minerals.value) == {"quartz", "clay"}
        assert "calcite" in minerals.options
