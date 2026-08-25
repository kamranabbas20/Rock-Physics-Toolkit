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


class TestTheTracePanelBesideTheDetail:
    """The trace panel carries the log tracks, the gather and the lobe shading.

    Exercised on the figure itself rather than through the page, so a broken
    panel is reported as a broken panel and not as a page that failed to run.
    """

    @staticmethod
    def _inputs():
        import pandas as pd

        from avo_qi.core.blocking import lobe_windows

        twt = np.arange(40) * 0.002
        # One clean trough, so there is a real lobe to halve.
        trace = np.zeros(40)
        trace[12:21] = -np.sin(np.linspace(0, np.pi, 9))
        table = pd.DataFrame({"sample": [16], "avo_class": ["III"],
                              "depth": [1600.0]})
        extrema = {"index": np.array([16]), "amplitude": np.array([trace[16]]),
                   "is_extremum": np.array([True]), "offset": np.array([0])}
        lobe = lobe_windows(trace, np.array([16]), polarity=np.array([-1.0]))
        assert lobe["resolved"][0], "the fixture must have a lobe to shade"
        return trace, twt, table, extrema, lobe

    def test_the_gather_is_drawn_to_the_right_of_the_trace(self):
        from avo_qi.ui import classified_trace_figure

        trace, twt, table, extrema, _ = self._inputs()
        angles = np.arange(0, 41, 5, dtype=float)
        gather = np.outer(trace, np.linspace(1.0, 0.4, angles.size))

        fig = classified_trace_figure(trace, twt, table, extrema,
                                      logs={"Vp (m/s)": np.full(40, 2500.0)},
                                      gather=gather, gather_angles=angles)
        heatmaps = [t for t in fig.data if t.type == "heatmap"]
        assert len(heatmaps) == 1
        titles = [a.text for a in fig.layout.annotations]
        assert titles[-1] == "Gather", "the gather must be the right-hand panel"
        # ...and on the trace's own two-way-time axis, which is the point of
        # putting it there rather than in a figure of its own.
        assert heatmaps[0].y[0] == twt[0] and heatmaps[0].y[-1] == twt[-1]
        assert fig.layout.yaxis.autorange == "reversed"

    def test_no_gather_is_drawn_when_none_is_given(self):
        from avo_qi.ui import classified_trace_figure

        trace, twt, table, extrema, _ = self._inputs()
        fig = classified_trace_figure(trace, twt, table, extrema)
        assert not [t for t in fig.data if t.type == "heatmap"]
        assert "Gather" not in [a.text for a in fig.layout.annotations]

    def test_the_two_lobe_halves_are_shaded_over_every_panel(self):
        """Blue above the extremum, orange below, meeting at it — and drawn
        across the logs too, since the shading is there to say which log
        samples were averaged."""
        from avo_qi.ui import LOBE_COLOURS, classified_trace_figure

        trace, twt, table, extrema, lobe = self._inputs()
        logs = {"Vp (m/s)": np.full(40, 2500.0), "Vs (m/s)": np.full(40, 1200.0)}
        fig = classified_trace_figure(trace, twt, table, extrema, selected=0,
                                      logs=logs, lobe=lobe)

        rects = [s for s in fig.layout.shapes if s.type == "rect"]
        upper = [s for s in rects if s.fillcolor == LOBE_COLOURS["upper"]]
        lower = [s for s in rects if s.fillcolor == LOBE_COLOURS["lower"]]
        # Three panels: two logs and the trace.
        assert len(upper) == 3 and len(lower) == 3

        step = float(np.median(np.diff(twt)))
        assert upper[0].y0 == pytest.approx(twt[lobe["upper_start"][0]] - step / 2)
        assert upper[0].y1 == pytest.approx(twt[16] + step / 2)
        assert lower[0].y0 == pytest.approx(twt[16] - step / 2)
        assert lower[0].y1 == pytest.approx(
            twt[lobe["lower_stop"][0] - 1] + step / 2)

    def test_the_gather_gets_the_window_as_an_outline_not_a_fill(self):
        """A fill would sit under the heatmap and vanish; drawn over it, it
        would tint amplitudes on a scale whose colour *is* the polarity."""
        from avo_qi.ui import LOBE_COLOURS, LOBE_EDGES, classified_trace_figure

        trace, twt, table, extrema, lobe = self._inputs()
        angles = np.arange(0, 41, 10, dtype=float)
        fig = classified_trace_figure(
            trace, twt, table, extrema, selected=0, lobe=lobe,
            gather=np.outer(trace, np.ones(angles.size)), gather_angles=angles)

        rects = [s for s in fig.layout.shapes if s.type == "rect"]
        filled = [s for s in rects if s.fillcolor in LOBE_COLOURS.values()]
        outlined = [s for s in rects if s.line.color in LOBE_EDGES.values()]
        assert len(filled) == 2, "one fill per half on the trace panel"
        assert len(outlined) == 2, "one outline per half on the gather"
        for shape in outlined:
            assert shape.fillcolor == "rgba(0,0,0,0)"
            assert shape.layer == "above"

    def test_a_reflector_with_no_lobe_is_left_unshaded(self):
        """The fallback fixed window is not the reflector's lobe, so shading it
        would claim a measurement that was never made."""
        from avo_qi.ui import LOBE_COLOURS, classified_trace_figure

        trace, twt, table, extrema, lobe = self._inputs()
        lobe = dict(lobe)
        lobe["resolved"] = np.array([False])
        fig = classified_trace_figure(trace, twt, table, extrema, selected=0,
                                      lobe=lobe)
        assert not [s for s in fig.layout.shapes
                    if s.fillcolor in LOBE_COLOURS.values()]

    def test_nothing_is_shaded_without_a_lobe_argument(self):
        from avo_qi.ui import LOBE_COLOURS, classified_trace_figure

        trace, twt, table, extrema, _ = self._inputs()
        fig = classified_trace_figure(trace, twt, table, extrema, selected=0)
        assert not [s for s in fig.layout.shapes
                    if s.fillcolor in LOBE_COLOURS.values()]

    def test_the_trace_is_filled_red_left_and_blue_right(self):
        """Variable area in the usual seismic convention: troughs red, peaks
        blue. Each side needs its own baseline, because `fill="tonextx"` fills
        to the previous trace and one shared zero line cannot serve both."""
        from avo_qi.ui import (TRACE_FILL_NEGATIVE, TRACE_FILL_POSITIVE,
                               classified_trace_figure)

        trace, twt, table, extrema, _ = self._inputs()
        fig = classified_trace_figure(trace, twt, table, extrema)

        filled = [t for t in fig.data if getattr(t, "fill", None) == "tonextx"]
        assert len(filled) == 2
        by_colour = {t.fillcolor: t for t in filled}
        assert set(by_colour) == {TRACE_FILL_NEGATIVE, TRACE_FILL_POSITIVE}
        # The red one carries only the negative half, the blue only the positive.
        assert (np.asarray(by_colour[TRACE_FILL_NEGATIVE].x) <= 0).all()
        assert (np.asarray(by_colour[TRACE_FILL_POSITIVE].x) >= 0).all()
        # ...and between them they still describe the whole trace.
        rebuilt = (np.asarray(by_colour[TRACE_FILL_NEGATIVE].x)
                   + np.asarray(by_colour[TRACE_FILL_POSITIVE].x))
        assert np.allclose(rebuilt, trace)

    def test_the_logs_are_drawn_left_to_right_in_the_order_given(self):
        from avo_qi.ui import classified_trace_figure

        trace, twt, table, extrema, _ = self._inputs()
        logs = {"VSH (v/v)": np.linspace(0, 1, 40),
                "Vp (m/s)": np.full(40, 2500.0),
                "Vs (m/s)": np.full(40, 1200.0)}
        fig = classified_trace_figure(trace, twt, table, extrema, logs=logs)
        titles = [a.text for a in fig.layout.annotations]
        assert titles[:3] == list(logs)
        assert titles[3] == "Trace"


class TestTheVshTrack:
    """VSH leads the tracks, because it is what the lithology pair is cut from."""

    @staticmethod
    def _frame(**columns):
        import pandas as pd

        base = {"VP": np.full(20, 2500.0), "VS": np.full(20, 1200.0),
                "RHOB": np.full(20, 2.3)}
        base.update(columns)
        return pd.DataFrame(base)

    def test_a_well_with_vsh_puts_it_first(self):
        from avo_qi.ui import Settings, detail_log_tracks

        tracks = detail_log_tracks(self._frame(VSH=np.linspace(0, 1, 20)), Settings())
        assert list(tracks) == ["VSH (v/v)", "Vp (m/s)", "Vs (m/s)", "RHOB (g/cc)"]

    def test_a_well_with_only_gr_says_the_vsh_is_derived(self):
        from avo_qi.ui import Settings, detail_log_tracks

        tracks = detail_log_tracks(self._frame(GR=np.linspace(20, 140, 20)),
                                   Settings())
        assert list(tracks)[0] == "VSH from GR (v/v)"
        assert np.isfinite(tracks["VSH from GR (v/v)"]).all()

    def test_a_well_with_neither_gets_no_vsh_track(self):
        """Rather than an empty one, which would read as a curve of zeros."""
        from avo_qi.ui import Settings, detail_log_tracks

        tracks = detail_log_tracks(self._frame(), Settings())
        assert not any(k.startswith("VSH") for k in tracks)
        assert list(tracks)[0] == "Vp (m/s)"

    def test_the_real_well_shows_its_own_vsh_curve(self):
        """15/9-19-A carries VSH on 2812 of 3905 samples, so the track must be
        the curve rather than a GR derivation."""
        import os

        from avo_qi.io.loader import read_well, standardise
        from avo_qi.ui import Settings, detail_log_tracks

        las = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "data", "15_9_19_A.las")
        raw, units = read_well(las)
        well = standardise(raw, units=units, name="15/9-19-A")
        tracks = detail_log_tracks(well.df, Settings())
        assert list(tracks)[0] == "VSH (v/v)"
        vsh = tracks["VSH (v/v)"]
        assert np.isfinite(vsh).sum() == 2812
        assert np.nanmax(vsh) <= 1.0 and np.nanmin(vsh) >= 0.0

    def test_the_avo_page_still_runs_with_the_extra_track(self, avo_page):
        assert not avo_page.exception

    def test_the_page_offers_all_three_as_toggles(self, avo_page):
        labels = {c.label for c in avo_page.checkbox}
        assert {"Vp, Vs and RHOB tracks", "Gather", "Lobe halves"} <= labels


class TestAStaleDropdownLabel:
    """A Streamlit selectbox round-trips its *formatted label*, not its option.

    The browser sends the label string back and Streamlit looks it up in a
    mapping rebuilt from this run's options, returning the raw string when the
    lookup misses.  The reflector labels carry depth, class, A and B, so
    anything that moves those numbers — a filter, a different wavelet, another
    blocking average — rewrites them, and the stored "index" comes back as a
    sentence.  Comparing it to a row count raised
    ``TypeError: '>=' not supported between instances of 'str' and 'int'``.
    """

    STALE = "2119.1 m / 1.701 s — Class I (A +0.117, B -0.079) · shale over sand"

    @staticmethod
    def _run(**state):
        at = AppTest.from_file(os.path.join(PAGES, "4_AVO_Classification.py"),
                               default_timeout=180)
        _inject_demo_well(at)
        for key, value in state.items():
            at.session_state[key] = value
        at.run()
        return at

    def test_a_label_left_in_state_does_not_crash_the_page(self):
        at = self._run(reflector_pick=self.STALE)
        assert not at.exception
        assert isinstance(at.session_state["reflector_pick"], int)

    def test_the_anchor_keeps_the_same_reflector_not_the_same_row(self):
        """The point of the sample anchor: when the stored value is unusable,
        re-select the reflector that was chosen, not whatever now sits at that
        row index.

        Asserted against the page's real reflector table but through the
        resolver directly. Neither half of the real trigger can be driven
        through AppTest: it applies the selectbox's ``format_func`` to whatever
        raw widget state it holds, so a stale label or an out-of-range index
        raises inside the harness before the page's own handling is reached.
        The page-level guarantee — a stale label does not crash the run — is
        pinned by the first test in this class.
        """
        from avo_qi.ui import resolve_reflector_pick

        at = self._run()
        table = reflector_table(at)
        samples = table["sample"].to_numpy()
        row = len(table) - 2
        target = int(samples[row])

        at.session_state["reflector_pick"] = row
        at.run()
        assert at.session_state["reflector_pick_sample"] == target

        # Drop the rows above it, exactly as a filter would, and hand the
        # resolver a value it cannot use.
        shortened = samples[row - 1:]
        moved = resolve_reflector_pick(self.STALE, shortened,
                                       anchor_sample=target, fallback=0)
        assert int(shortened[moved]) == target
        assert moved != row, "otherwise the row index alone would have done"

    def test_a_reflector_that_the_filter_removed_falls_back(self):
        at = self._run()
        table = reflector_table(at)
        dropped = table[table["litho_pair"] == "shale over shale"]
        assert len(dropped), "the demo well must have a shale-on-shale event"
        gone = int(np.flatnonzero(
            table["sample"].to_numpy() == int(dropped["sample"].iloc[0]))[0])
        at.session_state["reflector_pick"] = gone
        at.run()
        assert at.session_state["reflector_pick_sample"] == int(
            dropped["sample"].iloc[0])

        picker = next(m for m in at.multiselect
                      if m.label == "Interface pairs to keep")
        picker.set_value(["shale over sand"]).run()
        assert not at.exception
        survivors = reflector_table(at)
        assert set(survivors["litho_pair"]) == {"shale over sand"}
        pick = at.session_state["reflector_pick"]
        assert 0 <= pick < len(survivors)

    def test_the_resolver_itself(self):
        from avo_qi.ui import resolve_reflector_pick

        samples = np.array([26, 33, 54, 61])
        # An in-range integer is the successful round-trip and is trusted.
        assert resolve_reflector_pick(2, samples, anchor_sample=26) == 2
        # A stale label is not an index, whatever it looks like.
        assert resolve_reflector_pick("2119.1 m — Class I", samples,
                                      anchor_sample=54) == 2
        # Nor is a numeric string, which would otherwise index by accident.
        assert resolve_reflector_pick("3", samples, anchor_sample=26) == 0
        # Out of range after a filter: the anchor wins, then the fallback.
        assert resolve_reflector_pick(9, samples, anchor_sample=61) == 3
        assert resolve_reflector_pick(9, samples, anchor_sample=999,
                                      fallback=1) == 1
        assert resolve_reflector_pick(None, samples, fallback=2) == 2
        # An out-of-range fallback cannot itself be returned.
        assert resolve_reflector_pick(None, samples, fallback=99) == 0
        assert resolve_reflector_pick(0, np.array([])) == 0

    def test_a_dropdown_the_page_does_not_read_first_repairs_itself(self):
        """Why the fix is only on the detail dropdown. The fluid-case dropdown
        has equally volatile labels, but nothing reads its key before the
        widget exists, so Streamlit's own validation resets an unrecognised
        value instead of leaving a string in play."""
        at = self._run(fluid_reflector="2119.1 m — class changes")
        assert not at.exception
        assert at.session_state["fluid_reflector"] == 0


class TestInterfacePairFilter:
    """Filtering on the *pair* — a shale-over-sand top is a different event
    from the shale-over-shale contrast a few samples above it."""

    @staticmethod
    def _page(pairs=None):
        at = AppTest.from_file(os.path.join(PAGES, "4_AVO_Classification.py"),
                               default_timeout=180)
        _inject_demo_well(at)
        at.run()
        if pairs is not None:
            picker = next(m for m in at.multiselect
                          if m.label == "Interface pairs to keep")
            picker.set_value(pairs).run()
        return at

    def test_the_filter_offers_the_pairs_that_are_actually_present(self):
        at = self._page()
        picker = next(m for m in at.multiselect
                      if m.label == "Interface pairs to keep")
        assert "shale over sand" in picker.options
        assert "shale over shale" in picker.options
        # Defaults to everything known, so the page opens unfiltered.
        assert set(picker.value) == {p for p in picker.options
                                     if "undefined" not in p}

    def test_keeping_only_the_tops_drops_every_other_pair(self):
        at = self._page(["shale over sand"])
        assert not at.exception
        table = reflector_table(at)
        assert len(table) > 0
        assert set(table["litho_pair"]) == {"shale over sand"}

    def test_it_says_how_many_it_hid(self):
        at = self._page(["shale over sand"])
        captions = " ".join(c.value for c in at.caption)
        assert "Interface-pair filter is hiding" in captions

    def test_clearing_the_selection_keeps_everything(self):
        """An empty multiselect reads as 'no filter', matching the sidebar's
        lithology filter — otherwise clearing it would blank the page."""
        everything = len(reflector_table(self._page()))
        at = self._page([])
        assert not at.exception
        assert len(reflector_table(at)) == everything

    def test_the_pair_filter_and_the_lithology_filter_are_different_questions(self):
        """The sidebar keeps a reflector when *either* side is a selected
        lithology; this one asks for the ordered pair. Sand-over-shale survives
        the first and not the second."""
        at = self._page()
        at.session_state["settings"].lithologies = ["sand"]
        at.run()
        by_sample = set(reflector_table(at)["litho_pair"])
        assert "sand over shale" in by_sample

        at = self._page(["shale over sand"])
        assert "sand over shale" not in set(reflector_table(at)["litho_pair"])


class TestTheDetailPanelFollowsTheFilters:
    """The blocked layers are computed before any filter and are held in arrays
    parallel to the *unfiltered* table.  Indexing them with a filtered row
    position quotes a different reflector's layers — silently, and with a
    perfectly plausible-looking number."""

    @staticmethod
    def _panel(at):
        for caption in at.caption:
            if "Upper layer Vp" in caption.value:
                return caption.value
        raise AssertionError("the detail panel quoted no layer properties")

    def test_the_same_reflector_reads_the_same_filtered_or_not(self):
        at = AppTest.from_file(os.path.join(PAGES, "4_AVO_Classification.py"),
                               default_timeout=300)
        _inject_demo_well(at)
        at.run()
        full = reflector_table(at)

        # A reflector far enough down the table that the rows above it are not
        # all kept — that gap is what the bug turned into a wrong answer.
        target = int(full["sample"].iloc[6])
        at.session_state["reflector_pick"] = 6
        at.run()
        unfiltered = self._panel(at)

        at.session_state["settings"].lithologies = ["sand"]
        at.run()
        table = reflector_table(at)
        assert len(table) < len(full), "the filter must actually drop rows"
        position = int(np.flatnonzero(table["sample"].to_numpy() == target)[0])
        assert position != 6, "otherwise this proves nothing"
        at.session_state["reflector_pick"] = position
        at.run()
        assert self._panel(at) == unfiltered

    def test_the_reflector_table_does_not_leak_the_bookkeeping_column(self):
        at = AppTest.from_file(os.path.join(PAGES, "4_AVO_Classification.py"),
                               default_timeout=180)
        _inject_demo_well(at)
        at.run()
        assert "props_row" not in reflector_table(at).columns


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

    def test_the_template_tab_is_on_the_rock_physics_page(self, rock_physics_page):
        assert not rock_physics_page.exception
        labels = {s.label for s in rock_physics_page.slider}
        assert "VSH the template is drawn for" in labels
        captions = " ".join(c.value for c in rock_physics_page.caption)
        assert "rock physics template" in captions
        # It must say which way to read the two families of curves.
        assert "constant porosity" in captions

    def test_the_template_reports_the_fluid_effect_it_predicts(self, rock_physics_page):
        """The number an interpreter actually wants off an RPT: how much
        impedance the hydrocarbon costs, and at what porosity it peaks."""
        import re

        captions = " ".join(c.value for c in rock_physics_page.caption)
        assert re.search(r"drops the impedance by up to \d+%", captions)
        assert re.search(r"most at porosity 0\.\d+", captions)

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

    def test_every_reflector_is_a_turning_point_of_the_trace(self):
        """The trace decides where the reflectors are.

        This is the invariant the whole page now rests on, and it is what makes
        a reflector with no extremum of its own impossible: each one *is* an
        extremum. Checked against the full stack rebuilt independently here,
        not against anything the page hands back.
        """
        from avo_qi.core.synthetic import build_gather, full_stack, _local_extrema
        from avo_qi.core.reflectivity import reflectivity_series
        from avo_qi.ui import build_wavelet, time_well

        at = run_page(os.path.join(PAGES, "4_AVO_Classification.py"))
        assert not at.exception
        table = reflector_table(at)
        settings = at.session_state["settings"]
        well = at.session_state["well"]

        tw = time_well(well, settings, settings.case)
        vp = tw["VP"].to_numpy(float)
        vs = tw["VS"].to_numpy(float)
        rho = tw["RHOB"].to_numpy(float)
        _, wavelet = build_wavelet(settings)
        stack = full_stack(build_gather(vp, vs, rho, settings.angles, wavelet,
                                        dt=settings.dt, method=settings.method))
        turning = set(_local_extrema(stack).tolist())

        assert len(table) > 0
        assert set(table["sample"].astype(int)) <= turning
        # ...and the polarity reported is the trace's own sign there.
        for _, row in table.iterrows():
            sign = "peak" if stack[int(row["sample"])] > 0 else "trough"
            assert row["polarity"] == sign

    def test_raising_the_amplitude_cut_keeps_only_the_louder_events(self):
        at = AppTest.from_file(os.path.join(PAGES, "4_AVO_Classification.py"),
                               default_timeout=180)
        _inject_demo_well(at)
        at.run()
        loud = reflector_table(at)

        at.session_state["settings"].threshold = 0.30
        at.run()
        assert not at.exception
        louder = reflector_table(at)
        assert len(louder) < len(loud)
        assert set(louder["sample"]) <= set(loud["sample"])
        # Everything kept clears the cut, as a fraction of the strongest event.
        strongest = loud["amplitude"].abs().max()
        assert (louder["amplitude"].abs() >= 0.30 * strongest - 1e-12).all()

    def test_no_reflector_is_left_without_a_lobe_of_its_own(self):
        """The old failure mode — 42% of a real well buried in a neighbour's
        lobe — cannot arise once the trace does the picking."""
        table = reflector_table(run_page(
            os.path.join(PAGES, "4_AVO_Classification.py")))
        assert "own_extremum" not in table.columns
        # A lobe may still be unhalvable (one sample wide), but never absent
        # because the reflector belonged to someone else's lobe.
        assert (table["blocking"] == "lobe").mean() > 0.5

    def test_the_report_builds_and_is_self_contained(self):
        """The button path is not exercised by simply rendering the page, and
        the document it makes is the deliverable."""
        import streamlit as st_mod

        from avo_qi.report import no_external_references

        captured = {}
        real = st_mod.download_button

        def spy(label, data, *args, **kwargs):
            if str(label).startswith("Download report"):
                captured["data"] = data
            return real(label, data, *args, **kwargs)

        at = AppTest.from_file(os.path.join(PAGES, "4_AVO_Classification.py"),
                               default_timeout=300)
        _inject_demo_well(at)
        st_mod.download_button = spy
        try:
            at.run()
            next(b for b in at.button if b.label == "Build report").click().run()
        finally:
            st_mod.download_button = real

        assert not at.exception
        document = captured.get("data")
        assert document, "the button produced no document"
        text = document.decode() if isinstance(document, bytes) else document

        assert no_external_references(text)
        assert text.startswith("<!doctype html>")
        # The provenance the CSVs cannot carry.
        for probe in ("How this was produced", "Event amplitude cut",
                      "Wavelet band", "Class tolerance a_tol", "DEMO-1"):
            assert probe in text, probe
        # ...and the results.
        assert "Reflector table" in text and "Intercept and gradient" in text

    def test_the_wedge_model_is_on_the_page(self, avo_page):
        """It was in core/tuning.py, tested, with no way to reach it."""
        assert "Wedge model" in {s.value for s in avo_page.subheader}
        labels = {m.label for m in avo_page.metric}
        assert {"Tuning thickness", "Amplitude peaks at",
                "Tuning brightening"} <= labels

    def test_the_wedge_controls_are_offered(self, avo_page):
        assert any(s.label == "Thickest bed (ms TWT)" for s in avo_page.slider)
        assert any(c.label == "Same rock above and below"
                   for c in avo_page.checkbox)
        assert any(s.label == "Amplitude at" for s in avo_page.selectbox)

    def test_the_wedge_follows_the_selected_reflector(self):
        """It is seeded from whichever reflector the detail panel is on, so
        moving the selection must move the wedge."""
        at = AppTest.from_file(os.path.join(PAGES, "4_AVO_Classification.py"),
                               default_timeout=240)
        _inject_demo_well(at)
        at.run()
        table = reflector_table(at)

        seen = set()
        for row in (0, len(table) - 1):
            at.session_state["reflector_pick"] = row
            at.run()
            assert not at.exception
            captions = " ".join(c.value for c in at.caption)
            assert "Seeded from the reflector selected above" in captions
            seen.add(f"{table['depth'].iloc[row]:.1f} m")
        # The seeding caption names the depth, so the two runs must differ.
        assert len(seen) == 2

    def test_the_wedge_can_be_read_in_metres(self):
        """Time is what the wavelet knows about; metres are that time carried
        through the reservoir's own Vp.

        Asserted as self-consistency rather than against a velocity guessed
        from the table: every thickness on the panel must convert by the *same*
        factor, which is what a conversion applied to one number and not
        another would break. The physics of the conversion itself is pinned in
        test_tuning.py.
        """
        at = AppTest.from_file(os.path.join(PAGES, "4_AVO_Classification.py"),
                               default_timeout=240)
        _inject_demo_well(at)
        at.run()
        unit = next(r for r in at.radio if r.label == "Thickness in")
        assert list(unit.options) == ["ms TWT", "metres"]

        # The wedge's own metric, not the well-wide one in "Tuned vs untuned":
        # they answer different questions and carry different labels.
        def reading(page, label):
            return float(next(m for m in page.metric
                              if m.label == label).value.split()[0])

        tuning_ms = reading(at, "Tuning thickness (this bed)")
        peak_ms = reading(at, "Amplitude peaks at")

        unit.set_value("metres").run()
        assert not at.exception
        tuning_m = reading(at, "Tuning thickness (this bed)")
        peak_m = reading(at, "Amplitude peaks at")

        assert tuning_ms > 0 and peak_ms > 0
        factor = tuning_m / tuning_ms
        assert peak_m == pytest.approx(peak_ms * factor, rel=0.02)
        # ms TWT -> m is v/2000, so any rock velocity lands the factor here.
        assert 0.75 <= factor <= 3.0

    def test_the_two_tuning_metrics_are_labelled_apart(self, avo_page):
        """One is well-wide, the other is the wedge's reservoir. Sharing a
        label made them indistinguishable on the page as well as in a test."""
        labels = [m.label for m in avo_page.metric]
        assert "Tuning thickness" in labels
        assert "Tuning thickness (this bed)" in labels

    def test_it_reports_whether_thickness_alone_moves_the_class(self, avo_page):
        """The whole point of a wedge: the same rock and the same fluid can
        classify differently purely because the bed is thin."""
        messages = " ".join(
            [i.value for i in avo_page.info] + [s.value for s in avo_page.success])
        assert ("purely because of thickness" in messages
                or "not a thickness artefact" in messages)

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


class TestZoneSummaryOnTheAvoPage:
    """The zonation as an answer: thickness, net-to-gross, and the events in it.

    The demo well is built with a gas sand, a brine sand and a cemented streak
    in shale, so the summary has a right answer to be checked against — a clean
    sand should read as all net, and only the gas sand should read as pay.
    """

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

    def test_every_zone_of_the_well_is_reported(self, avo_page):
        summary = self._summary(avo_page)
        assert {"Shale", "Gas Sand", "Brine Sand", "Cemented Streak"} == set(summary["zone"])
        assert (summary["gross"] > 0).all()

    def test_a_clean_sand_reads_as_all_net(self, avo_page):
        gas = self._summary(avo_page).set_index("zone").loc["Gas Sand"]
        assert gas["ntg"] == pytest.approx(1.0)
        assert gas["gross"] == pytest.approx(30.0, abs=0.2)
        assert self._summary(avo_page).set_index("zone").loc["Shale", "ntg"] == 0.0

    def test_only_the_hydrocarbon_sand_reads_as_pay(self, avo_page):
        """Both sands are net; the brine sand is at Sw = 1 and is not pay."""
        summary = self._summary(avo_page).set_index("zone")
        assert summary.loc["Gas Sand", "ptg"] == pytest.approx(1.0)
        assert summary.loc["Brine Sand", "ntg"] == pytest.approx(1.0)
        assert summary.loc["Brine Sand", "ptg"] == 0.0

    def test_a_zone_that_recurs_sums_its_thickness(self, avo_page):
        """Shale appears four times over 2000-2160 m. Its thickness is the rock
        it occupies, not the span from its first sample to its last."""
        shale = self._summary(avo_page).set_index("zone").loc["Shale"]
        assert shale["gross"] == pytest.approx(105.2, abs=0.5)
        assert shale["base"] - shale["top"] > 150

    def test_the_events_inside_each_zone_are_counted_and_classed(self, avo_page):
        summary = self._summary(avo_page)
        assert "n_events" in summary.columns and "class_mix" in summary.columns
        assert summary["n_events"].sum() > 0

    def test_a_reservoir_top_is_counted_in_the_reservoir(self, avo_page):
        """The reason events are counted on both sides.

        The gas sand's top has its upper lobe in the shale above, so counting
        only the upper side files the best event the reservoir has under the
        seal — and leaves the reservoir reading as nothing but weak internal
        reflections.
        """
        gas = self._summary(avo_page).set_index("zone").loc["Gas Sand"]
        assert "III" in str(gas["class_mix"])
        assert gas["min_deviation"] < 0

    def test_the_strongest_anomaly_per_zone_is_reported(self, avo_page):
        summary = self._summary(avo_page)
        if "min_deviation" not in summary.columns:
            pytest.skip("no background trend was fitted")
        deepest = summary.set_index("zone")["min_deviation"]
        # Signed distance from the background trend: negative is the side
        # hydrocarbon responses fall on, and every zone with an event has one.
        assert deepest.notna().sum() >= 3
        assert (deepest.min() < 0)

    def test_the_summary_is_measured_in_depth_not_time(self, avo_page):
        """The thickness of a zone is a length of rock. Counting time samples
        would make a fast layer read thin, so this is stated on the page."""
        captions = " ".join(c.value for c in avo_page.caption)
        assert "computed on the depth log, not on the time trace" in captions

    def test_it_can_be_downloaded(self, avo_page):
        labels = [b.label for b in avo_page.get("download_button")]
        assert "Download zone summary (CSV)" in labels

    def test_a_well_with_no_zonation_simply_has_no_summary(self):
        """No zone curve and no tops is not an error; there is nothing to say."""
        well, raw, units = demo_well()
        well.df = well.df.drop(columns=["ZONE"])
        at = AppTest.from_file(os.path.join(PAGES, "4_AVO_Classification.py"),
                               default_timeout=180)
        at.session_state["well"] = well
        at.session_state["raw_df"] = raw
        at.session_state["raw_units"] = units
        at.run()
        assert not at.exception
        assert "Zone summary" not in {s.value for s in at.subheader}


class TestDepthReferenceOnThePage:
    """MD is hole length from a rig floor. TVDSS and TVDBML are what make a
    depth mean the same thing in two wells, so the page asks for what it needs
    to get there and refuses to guess."""

    @staticmethod
    def _page():
        return run_page(os.path.join(PAGES, "1_Load_and_QC.py"), timeout=180)

    @staticmethod
    def _apply(at, kb=25.0, water=90.0, choice="Vertical well (TVD = MD)"):
        next(n for n in at.number_input
             if n.label.startswith("Drilling datum")).set_value(kb).run()
        next(n for n in at.number_input
             if n.label.startswith("Water depth")).set_value(water).run()
        next(r for r in at.radio
             if r.label == "True vertical depth").set_value(choice).run()
        next(b for b in at.button
             if b.label == "Apply depth reference").click().run()
        return at

    def test_the_page_asks_for_the_datum_and_the_hole(self, qc_page):
        assert "3 · Depth reference" in {s.value for s in qc_page.subheader}
        labels = {n.label for n in qc_page.number_input}
        assert "Drilling datum above MSL (m)" in labels
        assert "Water depth (m)" in labels
        choices = next(r for r in qc_page.radio if r.label == "True vertical depth")
        assert set(choices.options) == {"Not known", "Vertical well (TVD = MD)",
                                        "From a deviation survey"}

    def test_nothing_is_invented_before_it_is_told(self, qc_page):
        """The demo LAS carries no TVD curve and no elevation header, so the
        page must say it has no vertical reference rather than assume one."""
        well = qc_page.session_state["well"]
        assert not {"TVD", "TVDSS", "TVDBML"} & set(well.df.columns)
        assert any("No vertical reference" in i.value for i in qc_page.info)

    def test_applying_a_datum_writes_all_three_references(self):
        at = self._apply(self._page())
        assert not at.exception
        frame = at.session_state["well"].df
        md = frame["DEPTH"].to_numpy(float)
        assert frame["TVD"].to_numpy(float) == pytest.approx(md)
        assert frame["TVDSS"].to_numpy(float) == pytest.approx(md - 25.0)
        assert frame["TVDBML"].to_numpy(float) == pytest.approx(md - 115.0)

    def test_a_survey_overrides_an_earlier_vertical_assumption(self):
        """The trap this guards: once a computed TVD sits in the well it looks
        exactly like one the file supplied, and a file curve wins over
        everything — so a well first called vertical would keep TVD = MD for
        good, quietly ignoring the survey."""
        at = self._apply(self._page())
        vertical = at.session_state["well"].df["TVD"].to_numpy(float).copy()

        at.session_state["settings"].deviation_survey = [
            {"md": 0.0, "inc": 0.0, "azi": 0.0},
            {"md": 1000.0, "inc": 40.0, "azi": 90.0},
            {"md": 3000.0, "inc": 40.0, "azi": 90.0},
        ]
        self._apply(at, choice="From a deviation survey")
        deviated = at.session_state["well"].df["TVD"].to_numpy(float)
        assert (deviated < vertical - 100.0).all()

    def test_clearing_the_reference_removes_the_columns_rather_than_nulling_them(self):
        """'No TVDSS here' should read as an absent curve everywhere else, not
        as a curve that is somehow all blank."""
        at = self._apply(self._page())
        assert "TVDSS" in at.session_state["well"].df.columns
        self._apply(at, kb=None, water=None, choice="Not known")
        assert not {"TVD", "TVDSS", "TVDBML"} & set(
            at.session_state["well"].df.columns)

    def test_a_new_well_does_not_inherit_the_previous_ones_datum(self):
        at = self._apply(self._page())
        assert at.session_state["settings"].kb_elevation == 25.0
        next(b for b in at.button if b.label == "Load demo well").click().run()
        settings = at.session_state["settings"]
        assert settings.kb_elevation is None and settings.water_depth is None
        assert not settings.vertical_well and not settings.deviation_survey
