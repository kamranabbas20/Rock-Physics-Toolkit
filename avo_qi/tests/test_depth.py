"""Depth references: MD to TVD, and TVD to the two datums.

The reason this exists at all is that MD is not comparable between wells. A
formation at 3800 m MD in one hole and 3800 m MD in another are not at the
same depth in the earth: the rig floors are at different heights, the water
depths differ, and one of the holes may be deviated. TVDSS and TVDBML are what
make a depth trend — of porosity, of velocity, of AVO class — mean anything
across wells.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from avo_qi.core.depth import (
    depth_references,
    minimum_curvature,
    survey_from_table,
    tvd_from_survey,
)


class TestMinimumCurvature:
    def test_a_vertical_hole_is_its_own_tvd(self):
        md = np.arange(0.0, 3000.0, 30.0)
        found = minimum_curvature(md, np.zeros(md.size), np.zeros(md.size))
        assert found["tvd"] == pytest.approx(md)
        assert found["north"] == pytest.approx(np.zeros(md.size))
        assert found["east"] == pytest.approx(np.zeros(md.size))

    def test_a_constant_angle_hole_matches_the_trigonometry(self):
        """A straight 30° hole drops cos(30°) of every metre drilled, and steps
        out sin(30°). No curvature, so the ratio factor must be exactly 1."""
        md = np.array([0.0, 100.0, 200.0, 300.0])
        found = minimum_curvature(md, np.full(4, 30.0), np.full(4, 90.0))
        assert found["tvd"] == pytest.approx(md * np.cos(np.radians(30.0)))
        assert found["east"] == pytest.approx(md * np.sin(np.radians(30.0)))
        assert found["north"] == pytest.approx(np.zeros(4), abs=1e-9)
        assert found["dogleg"][1:] == pytest.approx(np.zeros(3), abs=1e-9)

    def test_a_horizontal_hole_stops_going_down(self):
        found = minimum_curvature([0.0, 100.0], [90.0, 90.0], [0.0, 0.0])
        assert found["tvd"][1] == pytest.approx(0.0, abs=1e-9)
        assert found["north"][1] == pytest.approx(100.0)

    def test_the_arc_falls_between_the_two_tangents(self):
        """The point of minimum curvature: a build section is neither the hole
        angle at the top nor at the bottom, but an arc between them."""
        md = np.array([0.0, 100.0])
        arc = minimum_curvature(md, [0.0, 60.0], [0.0, 0.0])["tvd"][1]
        assert 100.0 * np.cos(np.radians(60.0)) < arc < 100.0
        # The balanced-tangential answer is the straight-segment average; the
        # arc drops slightly less than that.
        tangential = 50.0 * (np.cos(0.0) + np.cos(np.radians(60.0)))
        assert arc > tangential

    def test_the_dogleg_is_reported_in_degrees(self):
        found = minimum_curvature([0.0, 100.0], [0.0, 30.0], [0.0, 0.0])
        assert found["dogleg"][1] == pytest.approx(30.0)

    def test_bad_input_is_rejected(self):
        with pytest.raises(ValueError, match="same length"):
            minimum_curvature([0.0, 1.0], [0.0], [0.0, 0.0])
        with pytest.raises(ValueError, match="must increase"):
            minimum_curvature([100.0, 0.0], [0.0, 0.0], [0.0, 0.0])

    def test_an_empty_survey_is_not_an_error(self):
        assert minimum_curvature([], [], [])["tvd"].size == 0


class TestTvdFromSurvey:
    """Log samples are centimetres apart and survey stations tens of metres,
    so the samples have to be placed inside a survey interval."""

    def test_a_vertical_survey_leaves_the_log_alone(self):
        md = np.arange(1000.0, 1100.0, 0.15)
        tvd = tvd_from_survey(md, [0.0, 500.0, 1500.0], [0.0] * 3, [0.0] * 3)
        assert tvd == pytest.approx(md)

    def test_a_constant_angle_survey_scales_the_whole_log(self):
        md = np.arange(1000.0, 1100.0, 0.15)
        tvd = tvd_from_survey(md, [0.0, 2000.0], [40.0, 40.0], [90.0, 90.0])
        assert tvd == pytest.approx(md * np.cos(np.radians(40.0)))

    def test_samples_land_between_their_stations(self):
        """Monotonic, and bracketed by the station TVDs either side."""
        stations = np.array([0.0, 500.0, 1000.0, 1500.0])
        inc = np.array([0.0, 15.0, 45.0, 60.0])
        azi = np.array([0.0, 30.0, 30.0, 45.0])
        md = np.arange(0.0, 1501.0, 5.0)      # the grid must reach the last
        tvd = tvd_from_survey(md, stations, inc, azi)   # station to compare it
        assert (np.diff(tvd) > 0).all()
        at_stations = minimum_curvature(stations, inc, azi)["tvd"]
        for depth, expected in zip(stations, at_stations):
            assert tvd[np.argmin(np.abs(md - depth))] == pytest.approx(
                expected, abs=0.5)

    def test_tvd_never_exceeds_md_in_a_deviated_hole(self):
        md = np.arange(0.0, 4000.0, 10.0)
        tvd = tvd_from_survey(md, [0.0, 1000.0, 4000.0], [0.0, 0.0, 55.0],
                              [0.0, 120.0, 120.0])
        assert (tvd <= md + 1e-9).all()
        # ...and by a lot at the bottom: this is the error a vertical
        # assumption would make. The build starts at 1000 m and reaches 55°
        # only at total depth, so the average angle is well under 55°.
        assert md[-1] - tvd[-1] > 400.0

    def test_above_the_shallowest_station_the_hole_is_taken_as_vertical(self):
        tvd = tvd_from_survey([100.0, 200.0], [300.0, 900.0], [30.0, 30.0],
                              [0.0, 0.0])
        assert tvd[1] - tvd[0] == pytest.approx(100.0)

    def test_one_station_holds_its_angle(self):
        tvd = tvd_from_survey([1000.0, 1100.0], [500.0], [60.0], [0.0])
        assert tvd[1] - tvd[0] == pytest.approx(100.0 * np.cos(np.radians(60.0)))

    def test_unusable_stations_are_dropped_not_guessed(self):
        tvd = tvd_from_survey([0.0, 100.0],
                              [0.0, np.nan, 500.0], [0.0, 10.0, 0.0],
                              [0.0, 0.0, 0.0])
        assert tvd == pytest.approx([0.0, 100.0])
        with pytest.raises(ValueError, match="no usable stations"):
            tvd_from_survey([0.0], [np.nan], [np.nan], [np.nan])

    def test_an_unsorted_survey_is_sorted_rather_than_believed(self):
        md = np.arange(0.0, 1000.0, 10.0)
        jumbled = tvd_from_survey(md, [1000.0, 0.0], [30.0, 30.0], [0.0, 0.0])
        ordered = tvd_from_survey(md, [0.0, 1000.0], [30.0, 30.0], [0.0, 0.0])
        assert jumbled == pytest.approx(ordered)


class TestSurveyFromTable:
    def test_the_usual_column_names_are_recognised(self):
        for names in (("MD", "INC", "AZI"),
                      ("Depth", "Inclination", "Azimuth"),
                      ("MD (m)", "DEVI", "HAZI"),
                      ("SURVEY_MD", "DRIFT", "AZIM")):
            frame = pd.DataFrame({names[0]: [0.0, 100.0],
                                  names[1]: [0.0, 30.0],
                                  names[2]: [0.0, 90.0]})
            md, inc, azi = survey_from_table(frame)
            assert md == pytest.approx([0.0, 100.0])
            assert inc == pytest.approx([0.0, 30.0])

    def test_rows_with_gaps_are_dropped(self):
        frame = pd.DataFrame({"MD": [0.0, 100.0, 200.0],
                              "INC": [0.0, np.nan, 30.0],
                              "AZI": [0.0, 0.0, 90.0]})
        md, _, _ = survey_from_table(frame)
        assert md == pytest.approx([0.0, 200.0])

    def test_a_table_that_is_not_a_survey_says_so(self):
        with pytest.raises(ValueError) as err:
            survey_from_table(pd.DataFrame({"a": [1], "b": [2]}))
        assert "inclination" in str(err.value)


class TestDepthReferences:
    MD = np.array([3000.0, 3100.0, 3200.0])

    def test_the_datums_are_plain_subtractions(self):
        found = depth_references(self.MD, vertical=True, kb_elevation=25.0,
                                 water_depth=90.0)
        assert found["TVD"] == pytest.approx(self.MD)
        assert found["TVDSS"] == pytest.approx(self.MD - 25.0)
        assert found["TVDBML"] == pytest.approx(self.MD - 115.0)

    def test_everything_increases_downwards(self):
        """Stated once in the module and asserted here: subsea depth is
        positive below sea level, not a negative elevation."""
        found = depth_references(self.MD, vertical=True, kb_elevation=25.0,
                                 water_depth=90.0)
        for key in ("TVD", "TVDSS", "TVDBML"):
            assert (np.diff(found[key]) > 0).all()
            assert found[key][0] > 0

    def test_a_file_tvd_wins_over_a_survey(self):
        given = self.MD - 40.0
        found = depth_references(self.MD, tvd=given,
                                 survey=([0.0, 4000.0], [30.0, 30.0], [0.0, 0.0]))
        assert found["TVD"] == pytest.approx(given)
        assert found["source"] == "curve"

    def test_a_survey_wins_over_the_vertical_assumption(self):
        found = depth_references(self.MD, vertical=True,
                                 survey=([0.0, 4000.0], [40.0, 40.0], [0.0, 0.0]))
        assert found["source"] == "survey"
        assert found["TVD"] == pytest.approx(self.MD * np.cos(np.radians(40.0)))

    def test_a_survey_can_arrive_as_a_table(self):
        survey = pd.DataFrame({"MD": [0.0, 4000.0], "INC": [0.0, 0.0],
                               "AZI": [0.0, 0.0]})
        assert depth_references(self.MD, survey=survey)["TVD"] == pytest.approx(self.MD)

    def test_what_is_not_known_is_nan_and_said_out_loud(self):
        """The point of the whole function: a subsea depth invented from an
        assumed rig-floor height is worse than no subsea depth."""
        found = depth_references(self.MD)
        for key in ("TVD", "TVDSS", "TVDBML"):
            assert np.isnan(found[key]).all()
        assert found["source"] is None
        joined = " ".join(found["notes"])
        assert "No TVD" in joined and "No TVDSS" in joined and "No TVDBML" in joined

    def test_tvd_without_a_datum_gives_no_subsea_depth(self):
        found = depth_references(self.MD, vertical=True)
        assert found["TVD"] == pytest.approx(self.MD)
        assert np.isnan(found["TVDSS"]).all()
        assert np.isnan(found["TVDBML"]).all()

    def test_a_rig_floor_at_sea_level_is_a_number_not_a_missing_value(self):
        """`if not kb_elevation` would read 0 m as unknown."""
        found = depth_references(self.MD, vertical=True, kb_elevation=0.0,
                                 water_depth=0.0)
        assert found["TVDSS"] == pytest.approx(self.MD)
        assert found["TVDBML"] == pytest.approx(self.MD)

    def test_the_vertical_assumption_is_recorded_as_one(self):
        notes = " ".join(depth_references(self.MD, vertical=True)["notes"])
        assert "assumed" in notes.lower()


class TestTheLoaderRecognisesThem:
    """A file that already carries TVDSS should never have it recomputed."""

    @staticmethod
    def _frame():
        return pd.DataFrame({
            "DEPT": [3000.0, 3010.0], "TVD": [2900.0, 2909.0],
            "TVDSS": [2875.0, 2884.0], "TVDBML": [2785.0, 2794.0],
            "VP": [3000.0, 3100.0], "VS": [1500.0, 1550.0],
            "RHOB": [2.3, 2.35],
        })

    def test_each_reference_gets_its_own_column(self):
        from avo_qi.io.loader import guess_mnemonics, standardise

        mapping = guess_mnemonics(self._frame().columns)
        assert mapping["DEPTH"] == "DEPT"
        assert mapping["TVD"] == "TVD"
        assert mapping["TVDSS"] == "TVDSS"
        assert mapping["TVDBML"] == "TVDBML"
        well = standardise(self._frame())
        assert well.df["TVDSS"].iloc[0] == pytest.approx(2875.0)

    def test_a_bare_tvd_entry_does_not_swallow_the_subsea_curve(self):
        """Matching is by prefix as well as exactly, so ordering matters:
        `TVD` would otherwise claim `TVDSS` and leave the real TVD unmapped."""
        from avo_qi.io.loader import guess_mnemonics

        frame = self._frame().drop(columns=["TVD"])
        mapping = guess_mnemonics(frame.columns)
        assert mapping["TVDSS"] == "TVDSS"
        assert mapping.get("TVD") != "TVDSS"

    def test_they_are_converted_from_feet_like_the_measured_depth(self):
        from avo_qi.io.loader import standardise

        well = standardise(self._frame(), depth_unit="ft")
        assert well.df["TVDSS"].iloc[0] == pytest.approx(2875.0 / 3.280839895)
        assert any("TVDSS" in note and "ft to m" in note for note in well.notes)

    def test_a_las_without_elevations_reports_them_as_unknown(self):
        """15/9-19-A has EKB, EGL, KB and GL entries, all of them empty. The
        header cannot be trusted to carry them, which is why the UI asks."""
        import os

        from avo_qi.io.loader import read_las_header

        here = os.path.dirname(os.path.abspath(__file__))
        header = read_las_header(os.path.join(here, "data", "15_9_19_A.las"))
        assert header == {"kb_elevation": None, "ground_level": None,
                          "water_depth": None}

    def test_elevations_are_read_where_a_las_does_carry_them(self, tmp_path):
        import lasio

        from avo_qi.io.loader import read_las_header

        las = lasio.LASFile()
        las.well["NULL"] = lasio.HeaderItem("NULL", value=-999.25)
        las.params["EKB"] = lasio.HeaderItem("EKB", unit="M", value=25.4)
        las.params["EGL"] = lasio.HeaderItem("EGL", unit="M", value=0.0)
        las.params["WD"] = lasio.HeaderItem("WD", unit="M", value=91.0)
        las.append_curve("DEPT", np.array([3000.0, 3010.0]), unit="M")
        las.append_curve("VP", np.array([3000.0, 3100.0]), unit="M/S")
        path = tmp_path / "elev.las"
        las.write(str(path), version=2.0)

        header = read_las_header(str(path))
        assert header["kb_elevation"] == pytest.approx(25.4)
        assert header["water_depth"] == pytest.approx(91.0)
        assert header["ground_level"] == pytest.approx(0.0)


class TestCurvesFromTheFileWin:
    """A file's own TVDSS was made with the survey and the datum the people who
    drilled the well actually had. Nothing reconstructed here beats that."""

    MD = np.array([3000.0, 3100.0, 3200.0])

    def test_a_file_tvdss_is_not_overwritten(self):
        given = self.MD - 60.0
        found = depth_references(self.MD, tvd_ss=given, vertical=True,
                                 kb_elevation=25.0)
        assert found["TVDSS"] == pytest.approx(given)
        assert found["TVD"] == pytest.approx(self.MD)     # ...but TVD still resolves

    def test_tvd_can_be_reconstructed_from_a_subsea_curve(self):
        found = depth_references(self.MD, tvd_ss=self.MD - 60.0, kb_elevation=25.0)
        assert found["source"] == "subsea curve"
        assert found["TVD"] == pytest.approx(self.MD - 35.0)

    def test_a_file_tvdbml_is_not_overwritten(self):
        given = self.MD - 150.0
        found = depth_references(self.MD, tvd_bml=given, vertical=True,
                                 kb_elevation=25.0, water_depth=90.0)
        assert found["TVDBML"] == pytest.approx(given)

    def test_an_all_null_curve_counts_as_absent(self):
        """A TVDSS column of nulls is a column, not a datum."""
        found = depth_references(self.MD, tvd_ss=np.full(3, np.nan),
                                 vertical=True, kb_elevation=25.0)
        assert found["TVDSS"] == pytest.approx(self.MD - 25.0)
