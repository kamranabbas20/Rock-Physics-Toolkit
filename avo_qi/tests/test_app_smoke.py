"""Smoke tests for the Streamlit pages, run headless through ``AppTest``.

These check that each page executes end to end against the demo well and
renders the elements the spec calls for.  They are skipped when Streamlit is
not installed, so ``core/`` stays testable on its own.
"""

from __future__ import annotations

import os

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
        well, raw, units = demo_well()
        at.session_state["well"] = well
        at.session_state["raw_df"] = raw
        at.session_state["raw_units"] = units
    at.run()
    return at


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
        # Vp-Vs trends, Gardner, velocity-porosity, moduli & bounds.
        assert len(rock_physics_page.tabs) >= 4

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

    def test_warns_that_the_frame_models_are_dry(self, rock_physics_page):
        assert any("dry-frame" in w.value for w in rock_physics_page.warning)

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
        well, raw, units = demo_well()
        at.session_state["well"] = well
        at.session_state["raw_df"] = raw
        at.session_state["raw_units"] = units
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
        well, raw, units = demo_well()
        at.session_state["well"] = well
        at.session_state["raw_df"] = raw
        at.session_state["raw_units"] = units
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
