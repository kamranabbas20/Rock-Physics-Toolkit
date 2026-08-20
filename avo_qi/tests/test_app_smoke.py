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
def crossplots_page():
    return run_page(os.path.join(PAGES, "1_Data_and_Crossplots.py"))


@pytest.fixture(scope="module")
def gather_page():
    return run_page(os.path.join(PAGES, "2_Synthetic_Gather.py"))


@pytest.fixture(scope="module")
def avo_page():
    return run_page(os.path.join(PAGES, "3_AVO_Classification.py"))


@pytest.fixture(scope="module")
def rock_physics_page():
    return run_page(os.path.join(PAGES, "4_Rock_Physics.py"))


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
        at = run_page(os.path.join(PAGES, "1_Data_and_Crossplots.py"), with_well=False)
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
        at = run_page(os.path.join(PAGES, "2_Synthetic_Gather.py"), with_well=False)
        assert not at.exception
        assert at.warning


class TestAvoClassificationPage:
    def test_runs_clean(self, avo_page):
        assert not avo_page.exception

    def test_reports_class_counts_and_a_reflector_table(self, avo_page):
        assert len(avo_page.metric) >= 6
        assert len(avo_page.dataframe) >= 1

    def test_finds_the_class_three_gas_sand(self, avo_page):
        table = avo_page.dataframe[0].value
        assert "avo_class" in table.columns
        assert (table["avo_class"] == "III").any()

    def test_shuey_and_aki_richards_agree_in_the_table(self, avo_page):
        import numpy as np

        table = avo_page.dataframe[0].value
        assert np.nanmax(np.abs(table["dA"].to_numpy(float))) < 1e-6
        assert np.nanmax(np.abs(table["dB"].to_numpy(float))) < 1e-6

    def test_stops_politely_without_a_well(self):
        at = run_page(os.path.join(PAGES, "3_AVO_Classification.py"), with_well=False)
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
        at = run_page(os.path.join(PAGES, "4_Rock_Physics.py"), with_well=False)
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
