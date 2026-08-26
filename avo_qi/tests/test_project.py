"""The setup file: what it restores, and what it must never contain.

Two properties carry this feature. It has to restore *everything* you decided,
because a setup that silently dropped a field would rebuild most of a session
and look complete — the worst possible failure. And it must contain **no curve
values at all**, because the whole point of calling it a setup rather than a
project is that you can mail it to a colleague or commit it beside the code
without mailing them proprietary logs.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from avo_qi.project import (EXCLUDED, PER_WELL, SAVED, SCHEMA,  # noqa: E402
                            apply_setup, build_setup, compare_fingerprint,
                            describe, dumps, fingerprint, loads,
                            suggested_name)


def configured():
    """A Settings with every saved field moved off its default."""
    from avo_qi.ui import Settings

    s = Settings()
    s.dt = 0.002
    s.t0 = 2.4
    s.angle_min, s.angle_max, s.angle_step = 2.0, 36.0, 3.0
    s.method = "aki_richards"
    s.a_tol = 0.035
    s.wavelet_kind = "Ormsby"
    s.ricker_freq = 25.0
    s.ormsby = (4.0, 9.0, 55.0, 70.0)
    s.wavelet_length = 0.2
    s.threshold = 0.08
    s.case = "gas"
    s.zones = ["Brent"]
    s.zone_names = {"1": "Brent"}
    s.zone_tops = [{"zone": "Brent", "top": 3800.0}]
    s.kb_elevation = 25.0
    s.water_depth = 90.0
    s.deviation_survey = [{"md": 0.0, "inc": 0.0, "azi": 0.0},
                          {"md": 3000.0, "inc": 41.0, "azi": 120.0}]
    s.vertical_well = False
    s.petrophysics = {"mode": "fill", "rw": 0.04, "m": 1.9}
    s.vs_prediction = {"model": "greenberg_castagna", "degree": 1}
    s.fluid_model = {"salinity": 35000.0, "api": 32.0}
    s.case_mapping = {"gas": {"VP": "VP_G"}}
    s.vsh_cutoffs = {"sand": 0.12, "silty sand": 0.3, "silt": 0.55}
    s.net_cutoffs = {"vsh": 0.3, "phi": 0.1, "sw": 0.6}
    s.net_pay = True
    s.lithologies = ["sand", "silty sand"]
    s.gr_method = "larionov_tertiary"
    s.near, s.mid, s.far = (0.0, 10.0), (11.0, 24.0), (25.0, 36.0)
    return s


@pytest.fixture(scope="module")
def well():
    from test_multi_well import second_well

    return second_well()[0]


class TestEveryFieldIsAccountedFor:
    def test_no_setting_is_silently_forgotten(self):
        """The test that keeps this feature honest as the app grows: a new
        Settings field must be saved or explicitly excluded, never neither."""
        from avo_qi.ui import Settings

        fields = {f.name for f in Settings.__dataclass_fields__.values()}
        unaccounted = fields - set(SAVED) - set(EXCLUDED)
        assert not unaccounted, (
            "add these to project.SAVED or project.EXCLUDED: %s"
            % sorted(unaccounted))

    def test_nothing_is_claimed_that_does_not_exist(self):
        from avo_qi.ui import Settings

        fields = {f.name for f in Settings.__dataclass_fields__.values()}
        assert set(SAVED) <= fields
        assert set(EXCLUDED) <= fields

    def test_every_exclusion_gives_a_reason(self):
        assert all(isinstance(why, str) and len(why) > 20
                   for why in EXCLUDED.values())

    def test_the_per_well_list_is_the_one_the_app_swaps(self):
        """If these drifted apart, switching wells and saving a setup would
        disagree about which settings belong to a hole."""
        from avo_qi.ui import WELL_SETTINGS

        assert set(PER_WELL) == set(WELL_SETTINGS)


class TestItContainsNoWellData:
    def test_no_curve_values_reach_the_file(self, well):
        """The property that makes it shareable. Every value in the well's
        curves, checked against every number in the file."""
        setup = build_setup(well, configured())
        text = dumps(setup).decode()

        for curve in ("VP", "VS", "RHOB", "GR", "VSH"):
            values = well.df[curve].to_numpy(float)
            values = values[np.isfinite(values)][:400]
            for value in values:
                assert repr(round(float(value), 4)) not in text, curve

    def test_the_fingerprint_names_curves_but_never_their_values(self, well):
        found = fingerprint(well)
        assert "VP" in found["curves"] and "VS" in found["curves"]
        assert set(found) == {"well", "samples", "depth_min", "depth_max",
                              "curves"}

    def test_it_stays_small_enough_to_mail(self, well):
        assert len(dumps(build_setup(well, configured()))) < 8_000

    def test_it_is_readable_text(self, well):
        """Not a pickle: someone should be able to open it and see what a
        colleague decided, and diff two of them."""
        payload = dumps(build_setup(well, configured()))
        parsed = json.loads(payload.decode())
        assert parsed["kind"] == "avo-qi-setup"
        assert parsed["settings"]["kb_elevation"] == 25.0


class TestRoundTrip:
    def test_every_saved_field_comes_back(self, well):
        from avo_qi.ui import Settings

        original = configured()
        restored = Settings()
        setup = loads(dumps(build_setup(well, original)))
        apply_setup(setup, restored)

        for name in SAVED:
            assert getattr(restored, name) == getattr(original, name), name

    def test_it_reports_what_it_changed(self, well):
        from avo_qi.ui import Settings

        blank = Settings()
        changed = apply_setup(loads(dumps(build_setup(well, configured()))),
                              blank)
        assert "kb_elevation" in changed and "zone_tops" in changed
        # Applying the same setup twice changes nothing the second time.
        assert apply_setup(loads(dumps(build_setup(well, configured()))),
                           blank) == []

    def test_integer_dict_keys_survive(self, well):
        """The bug the browser round-trip caught. ``zone_names`` is keyed by
        the numeric code in the ZONE curve, and JSON has no integer keys — so
        a plain dump turned {1: "Shale"} into {"1": "Shale"}, every lookup by
        code missed, and the app came back showing bare codes and the wrong
        interval count. The file itself looked perfectly fine."""
        from avo_qi.sample_data.make_demo_well import ZONE_NAMES
        from avo_qi.ui import Settings

        original = Settings()
        original.zone_names = dict(ZONE_NAMES)
        assert all(isinstance(k, int) for k in original.zone_names)

        restored = Settings()
        apply_setup(loads(dumps(build_setup(well, original))), restored)
        assert restored.zone_names == original.zone_names
        assert all(isinstance(k, int) for k in restored.zone_names)

    def test_a_change_of_key_type_alone_counts_as_a_change(self):
        """Reporting what changed must compare the values as they really are.
        Comparing two JSON-safe copies would call {1: "a"} and {"1": "a"}
        equal — exactly the difference that breaks a zone lookup."""
        from avo_qi.ui import Settings

        restored = Settings()
        restored.zone_names = {"1": "Shale"}
        setup = {"kind": "avo-qi-setup", "schema": SCHEMA,
                 "settings": {"zone_names": {"__pairs__": [[1, "Shale"]]}}}
        assert apply_setup(setup, restored) == ["zone_names"]
        assert restored.zone_names == {1: "Shale"}

    def test_string_keyed_dicts_stay_plain_in_the_file(self, well):
        """The envelope is only used where it is needed, so the file stays
        readable for everything else."""
        text = dumps(build_setup(well, configured())).decode()
        assert '"vsh_cutoffs": {\n      "sand"' in text
        assert "__pairs__" not in json.loads(text)["settings"]["vsh_cutoffs"]

    def test_tuples_survive_the_json_round_trip(self, well):
        """JSON has no tuples, and the angle stacks are compared as tuples."""
        from avo_qi.ui import Settings

        restored = Settings()
        apply_setup(loads(dumps(build_setup(well, configured()))), restored)
        assert isinstance(restored.ormsby, tuple)
        assert isinstance(restored.near, tuple)

    def test_the_angles_property_still_works_afterwards(self, well):
        from avo_qi.ui import Settings

        restored = Settings()
        apply_setup(loads(dumps(build_setup(well, configured()))), restored)
        assert restored.angles[0] == pytest.approx(2.0)
        assert restored.angles[-1] <= 36.0

    def test_only_the_fields_asked_for_are_applied(self, well):
        from avo_qi.ui import Settings

        restored = Settings()
        apply_setup(loads(dumps(build_setup(well, configured()))), restored,
                    fields=("kb_elevation",))
        assert restored.kb_elevation == 25.0
        assert restored.water_depth is None          # untouched


class TestItRefusesWhatItShould:
    def test_a_file_that_is_not_json(self):
        with pytest.raises(ValueError, match="not a setup file"):
            loads(b"KB 25.0\nWD 90.0\n")

    def test_some_other_json_file(self):
        with pytest.raises(ValueError, match="not an AVO & QI setup file"):
            loads(json.dumps({"kind": "something-else"}))

    def test_a_setup_from_a_newer_build(self):
        with pytest.raises(ValueError, match="newer version"):
            loads(json.dumps({"kind": "avo-qi-setup", "schema": SCHEMA + 1}))

    def test_a_setup_with_no_schema(self):
        with pytest.raises(ValueError, match="which schema"):
            loads(json.dumps({"kind": "avo-qi-setup"}))

    def test_an_older_setup_leaves_newer_settings_alone(self, well):
        """Forward compatibility the useful way round: a file that predates a
        field must not blank it."""
        from avo_qi.ui import Settings

        setup = loads(dumps(build_setup(well, configured())))
        setup["settings"].pop("net_cutoffs")
        restored = Settings()
        before = dict(restored.net_cutoffs)
        apply_setup(setup, restored)
        assert restored.net_cutoffs == before
        assert restored.kb_elevation == 25.0        # the rest still applied


class TestRecognisingTheWell:
    def test_the_well_it_was_built_for_matches(self, well):
        setup = build_setup(well, configured())
        assert compare_fingerprint(setup, well)["matches"]

    def test_a_different_well_is_noticed(self, well):
        from test_app_smoke import demo_well

        setup = build_setup(well, configured())
        found = compare_fingerprint(setup, demo_well()[0])
        assert not found["matches"]
        assert any("15/9-19-A" in d for d in found["differences"])
        assert any("samples" in d for d in found["differences"])

    def test_a_shortened_well_is_noticed(self, well):
        import copy

        setup = build_setup(well, configured())
        trimmed = copy.copy(well)
        trimmed.df = well.df.iloc[500:].reset_index(drop=True)
        found = compare_fingerprint(setup, trimmed)
        assert not found["matches"]
        assert any("top at" in d for d in found["differences"])

    def test_a_missing_curve_is_noticed(self, well):
        import copy

        setup = build_setup(well, configured())
        stripped = copy.copy(well)
        stripped.df = well.df.drop(columns=["VS"])
        found = compare_fingerprint(setup, stripped)
        assert any("VS" in d for d in found["differences"])

    def test_an_extra_curve_is_not_held_against_it(self, well):
        """Re-exporting a well with two more curves is ordinary; only the
        person doing it knows whether it is the same well, so a difference is
        reported and never enforced."""
        import copy

        setup = build_setup(well, configured())
        richer = copy.copy(well)
        richer.df = well.df.assign(EXTRA=1.0)
        assert compare_fingerprint(setup, richer)["matches"]


class TestPresentingIt:
    def test_the_summary_says_what_is_inside(self, well):
        found = describe(build_setup(well, configured(), note="post-tie"))
        assert found["well"] == "15/9-19-A"
        assert found["tops"] == 1 and found["survey_stations"] == 2
        assert found["datum"] == 25.0
        assert found["vs_model"] == "greenberg_castagna"
        assert found["petrophysics"] == "fill"
        assert found["note"] == "post-tie"
        assert found["fields"] == len(SAVED)

    def test_the_filename_names_the_well(self, well):
        assert suggested_name(well).startswith("15_9-19-A")
        assert suggested_name(well).endswith(".avoqi-setup.json")

    def test_an_awkward_well_name_still_gives_a_filename(self):
        class Bare:
            name = "../../etc/passwd"
            df = None

        found = suggested_name(Bare())
        assert "/" not in found and ".." not in found.split(".avoqi")[0]


class TestThroughThePage:
    """The setup file where it is actually used: saved from one session,
    applied in another."""

    @staticmethod
    def page(settings=None, upload=None):
        import os

        from streamlit.testing.v1 import AppTest

        from avo_qi.sample_data.make_demo_well import ZONE_NAMES
        from avo_qi.ui import Settings
        from test_app_smoke import PAGES, demo_well

        demo, raw, units = demo_well()
        at = AppTest.from_file(os.path.join(PAGES, "1_Load_and_QC.py"),
                               default_timeout=300)
        at.session_state["well"] = demo
        at.session_state["raw_df"] = raw
        at.session_state["raw_units"] = units
        chosen = settings or Settings()
        chosen.zone_names = dict(ZONE_NAMES)
        at.session_state["settings"] = chosen
        at.run()
        return at

    def test_the_page_offers_a_setup_file(self):
        at = self.page()
        assert not at.exception
        assert "10 · Setup file" in {s.value for s in at.subheader}
        assert any("setup" in b.label.lower() for b in at.download_button)

    def test_a_setup_built_from_a_live_session_carries_its_settings(self):
        """AppTest cannot read a download button's bytes, so this checks the
        real integration point instead: the settings object the page has been
        working with, handed to the same builder the page calls."""
        from avo_qi.ui import Settings

        chosen = Settings()
        chosen.kb_elevation = 31.0
        chosen.zone_tops = [{"zone": "Top Sand", "top": 2040.0}]
        at = self.page(settings=chosen)

        setup = build_setup(at.session_state["well"],
                            at.session_state["settings"])
        assert setup["settings"]["kb_elevation"] == 31.0
        assert setup["settings"]["zone_tops"][0]["zone"] == "Top Sand"
        assert setup["well"]["well"] == "DEMO-1"

    def test_a_setup_saved_from_one_session_restores_another(self):
        from avo_qi.ui import Settings

        chosen = Settings()
        chosen.kb_elevation = 31.0
        chosen.water_depth = 120.0
        chosen.threshold = 0.09
        chosen.vs_prediction = {"model": "mudrock"}
        at = self.page(settings=chosen)
        payload = dumps(build_setup(at.session_state["well"],
                                    at.session_state["settings"]))

        fresh = Settings()
        apply_setup(loads(payload), fresh)
        assert fresh.kb_elevation == 31.0
        assert fresh.water_depth == 120.0
        assert fresh.threshold == pytest.approx(0.09)
        assert fresh.vs_prediction == {"model": "mudrock"}

    def test_the_page_and_the_module_agree_on_what_is_saved(self):
        """The page tells the user how many fields the file holds. If that
        number came from anywhere but the module it would drift."""
        at = self.page()
        text = " ".join(c.value for c in at.caption)
        assert "%d fields" % len(SAVED) in text

    def test_the_page_says_it_contains_no_logs(self):
        at = self.page()
        text = " ".join(c.value for c in at.caption)
        assert "no curve values" in text


class TestApplyingASetupMovesTheControls:
    """The trap Streamlit sets, and the reason the setup looked inert.

    A keyed widget's own state outranks its ``value=`` argument. So writing
    ``settings.kb_elevation = 31`` does nothing visible — the control still
    shows what it showed, and on the next run writes its stale value straight
    back over the setting. The fix is to forget the widget's key so it
    re-reads; these tests hold the bookkeeping that makes that possible.
    """

    @staticmethod
    def page_source():
        import os

        from test_app_smoke import PAGES

        text = ""
        for name in ("1_Load_and_QC.py",):
            with open(os.path.join(PAGES, name)) as handle:
                text += handle.read()
        import avo_qi.ui as ui
        with open(ui.__file__) as handle:
            text += handle.read()
        return text

    def test_every_listed_key_still_exists_in_the_app(self):
        """A renamed widget would otherwise leave a key that clears nothing,
        and the setup would go quietly back to being inert."""
        from avo_qi.project import WIDGET_KEYS

        source = self.page_source()
        for setting, keys in WIDGET_KEYS.items():
            for key in keys:
                assert 'key="%s"' % key in source, "%s -> %s" % (setting, key)

    def test_it_only_names_settings_that_are_saved(self):
        from avo_qi.project import WIDGET_KEYS

        assert set(WIDGET_KEYS) <= set(SAVED)

    def test_only_the_changed_settings_forget_their_widgets(self):
        """Clearing every key on every apply would throw away controls the
        setup did not touch."""
        from avo_qi.project import stale_widget_keys

        assert stale_widget_keys(["kb_elevation"]) == ["kb_elevation_input"]
        assert stale_widget_keys([]) == []
        assert "cut_sand" in stale_widget_keys(["vsh_cutoffs"])
        # A setting with no widget behind it contributes nothing.
        assert stale_widget_keys(["zone_tops"]) == []

    def test_the_datum_control_reads_the_setting_it_mirrors(self):
        """Forgetting the key only helps if the widget's default comes from
        the setting. If someone rewrote it to a literal, this breaks."""
        import os

        from test_app_smoke import PAGES

        with open(os.path.join(PAGES, "1_Load_and_QC.py")) as handle:
            source = handle.read()
        assert "_kb_default = (settings.kb_elevation" in source
        assert "_wd_default = (settings.water_depth" in source
