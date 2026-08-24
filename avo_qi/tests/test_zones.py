"""Zonation from a discrete LAS curve or a tops list."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from avo_qi.core.zones import (
    UNZONED,
    ZONE_MNEMONICS,
    assign_zones,
    zone_of_interface,
    zones_from_curve,
    zones_from_tops,
    zone_of_lobe,
)

STEP = 0.1524
NAMES = {1: "Shale A", 2: "Sand A", 3: "Sand B"}


def zoned_well(null_top=20):
    """Depth and a zone curve where Sand A appears twice, with a null top."""
    depth = np.arange(2000.0, 2160.0, STEP)
    codes = np.full(depth.size, 1.0)
    codes[(depth >= 2040) & (depth < 2070)] = 2
    codes[(depth >= 2090) & (depth < 2105)] = 3
    codes[(depth >= 2120) & (depth < 2130)] = 2      # Sand A again, deeper
    codes[:null_top] = np.nan
    return depth, codes


class TestZonesFromCurve:
    def test_every_interval_is_found(self):
        depth, codes = zoned_well()
        zones = zones_from_curve(depth, codes, NAMES)
        assert list(zones["zone"]) == ["Shale A", "Sand A", "Shale A", "Sand B",
                                       "Shale A", "Sand A", "Shale A"]

    def test_a_zone_that_reappears_stays_a_separate_interval(self):
        """Merging the two Sand A intervals would invent a 50 m sand."""
        depth, codes = zoned_well()
        zones = zones_from_curve(depth, codes, NAMES)
        sand = zones[zones["zone"] == "Sand A"]
        assert len(sand) == 2
        assert sand["top"].iloc[1] > sand["base"].iloc[0]

    def test_intervals_meet_without_gaps(self):
        depth, codes = zoned_well()
        zones = zones_from_curve(depth, codes, NAMES)
        assert np.allclose(zones["base"].to_numpy()[:-1], zones["top"].to_numpy()[1:])

    def test_unmapped_codes_keep_their_own_value_as_a_name(self):
        depth, codes = zoned_well()
        zones = zones_from_curve(depth, codes)          # no names given
        assert set(zones["zone"]) == {"1", "2", "3"}

    def test_null_samples_fall_outside_every_zone(self):
        depth, codes = zoned_well(null_top=20)
        zones = zones_from_curve(depth, codes, NAMES)
        assert zones["top"].iloc[0] == pytest.approx(depth[20])

    def test_thin_flickers_can_be_dropped(self):
        depth, codes = zoned_well()
        codes[500] = 9                                   # a one-sample flicker
        assert 9 in set(zones_from_curve(depth, codes)["code"])
        cleaned = zones_from_curve(depth, codes, min_samples=3)
        assert 9 not in set(cleaned["code"])

    def test_non_integer_codes_survive(self):
        depth = np.arange(0.0, 10.0, 1.0)
        codes = np.array([1.5] * 5 + [2.5] * 5)
        zones = zones_from_curve(depth, codes)
        assert len(zones) == 2

    def test_string_codes_work(self):
        depth = np.arange(0.0, 6.0, 1.0)
        codes = np.array(["A", "A", "B", "B", "B", "A"], dtype=object)
        zones = zones_from_curve(depth, codes)
        assert list(zones["zone"]) == ["A", "B", "A"]

    def test_empty_and_mismatched_input(self):
        assert zones_from_curve(np.array([]), np.array([])).empty
        with pytest.raises(ValueError):
            zones_from_curve(np.arange(5.0), np.arange(3.0))


class TestAssignZones:
    def test_every_sample_below_the_nulls_gets_a_zone(self):
        depth, codes = zoned_well(null_top=20)
        labels = assign_zones(depth, zones_from_curve(depth, codes, NAMES))
        assert int(np.sum(labels == UNZONED)) == 20

    def test_the_deepest_sample_is_inside_its_zone(self):
        """The base of an interval is exclusive, so the last sample used to
        fall outside the zone it belongs to."""
        depth, codes = zoned_well(null_top=0)
        labels = assign_zones(depth, zones_from_curve(depth, codes, NAMES))
        assert labels[-1] != UNZONED
        assert not (labels == UNZONED).any()

    def test_labels_match_the_intervals(self):
        depth, codes = zoned_well(null_top=0)
        labels = assign_zones(depth, zones_from_curve(depth, codes, NAMES))
        sand = (depth >= 2040) & (depth < 2070)
        assert set(labels[sand]) == {"Sand A"}

    def test_no_zones_leaves_everything_unzoned(self):
        depth = np.arange(10.0)
        assert set(assign_zones(depth, None)) == {UNZONED}
        assert set(assign_zones(depth, pd.DataFrame())) == {UNZONED}

    def test_an_open_base_runs_to_the_end(self):
        zones = pd.DataFrame({"zone": ["A"], "top": [5.0], "base": [np.nan]})
        labels = assign_zones(np.arange(10.0), zones)
        assert set(labels[5:]) == {"A"}
        assert set(labels[:5]) == {UNZONED}


class TestZonesFromTops:
    def test_each_zone_runs_to_the_next_top(self):
        zones = zones_from_tops({"A": 2000.0, "B": 2040.0, "C": 2070.0},
                                base_depth=2160.0)
        assert list(zones["zone"]) == ["A", "B", "C"]
        assert zones["base"].iloc[0] == pytest.approx(2040.0)
        assert zones["base"].iloc[-1] == pytest.approx(2160.0)

    def test_tops_are_sorted_by_depth(self):
        zones = zones_from_tops({"C": 2070.0, "A": 2000.0, "B": 2040.0})
        assert list(zones["zone"]) == ["A", "B", "C"]

    def test_the_deepest_base_is_open_without_a_base_depth(self):
        zones = zones_from_tops({"A": 2000.0, "B": 2040.0})
        assert np.isnan(zones["base"].iloc[-1])

    def test_a_dataframe_of_tops_works(self):
        frame = pd.DataFrame({"Formation": ["A", "B"], "MD": [2000.0, 2050.0]})
        zones = zones_from_tops(frame, base_depth=2100.0)
        assert list(zones["zone"]) == ["A", "B"]

    def test_missing_columns_are_reported_clearly(self):
        with pytest.raises(ValueError) as err:
            zones_from_tops(pd.DataFrame({"a": [1], "b": [2]}))
        assert "name column" in str(err.value)

    def test_non_numeric_depths_are_dropped(self):
        frame = pd.DataFrame({"zone": ["A", "B"], "top": [2000.0, "nonsense"]})
        assert len(zones_from_tops(frame)) == 1


class TestZoneOfInterface:
    def test_a_zone_boundary_is_flagged(self):
        depth, codes = zoned_well(null_top=0)
        zones = zones_from_curve(depth, codes, NAMES)
        labels = assign_zones(depth, zones)
        boundary = int(zones["last"].iloc[0])            # last sample of Shale A
        found = zone_of_interface(labels, [boundary, boundary + 40])
        assert found["is_zone_boundary"][0]
        assert found["zone"][0] == "Shale A"
        assert found["zone_below"][0] == "Sand A"
        assert not found["is_zone_boundary"][1]          # well inside Sand A

    def test_the_last_interface_clips(self):
        labels = np.array(["A", "A", "B"], dtype=object)
        found = zone_of_interface(labels, [2])
        assert found["zone_below"][0] == "B"

    def test_empty_labels_are_safe(self):
        found = zone_of_interface(np.array([], dtype=object), [0, 1])
        assert found["zone"].size == 0


class TestZoneCurveInALas:
    """A zone curve must be recognised on load and must not be interpolated."""

    def test_the_loader_recognises_a_zone_mnemonic(self, tmp_path):
        import lasio

        from avo_qi.io.loader import read_well, standardise

        depth, codes = zoned_well(null_top=0)
        las = lasio.LASFile()
        las.well["NULL"] = lasio.HeaderItem("NULL", value=-999.25)
        las.append_curve("DEPT", depth, unit="M")
        las.append_curve("VP", np.full(depth.size, 2400.0), unit="M/S")
        las.append_curve("VS", np.full(depth.size, 1200.0), unit="M/S")
        las.append_curve("RHOB", np.full(depth.size, 2.35), unit="G/C3")
        las.append_curve("ZONE", codes, unit="")
        path = str(tmp_path / "zoned.las")
        las.write(path, version=2.0, fmt="%.4f")

        raw, units = read_well(path)
        well = standardise(raw, units=units)
        assert well.mapping.get("ZONE") == "ZONE"
        assert "ZONE" in well.df.columns

    @pytest.mark.parametrize("mnemonic", ["ZONE", "FORMATION", "MARKER", "UNIT"])
    def test_common_zone_mnemonics_are_listed(self, mnemonic):
        assert mnemonic in ZONE_MNEMONICS

    def test_zone_codes_stay_discrete_through_the_round_trip(self, tmp_path):
        """Zone codes are labels, not measurements — averaging them would
        invent zones that do not exist."""
        import lasio

        from avo_qi.io.loader import read_well, standardise

        depth, codes = zoned_well(null_top=0)
        las = lasio.LASFile()
        las.append_curve("DEPT", depth, unit="M")
        las.append_curve("VP", np.full(depth.size, 2400.0), unit="M/S")
        las.append_curve("VS", np.full(depth.size, 1200.0), unit="M/S")
        las.append_curve("RHOB", np.full(depth.size, 2.35), unit="G/C3")
        las.append_curve("ZONE", codes, unit="")
        path = str(tmp_path / "z.las")
        las.write(path, version=2.0, fmt="%.4f")

        raw, units = read_well(path)
        loaded = standardise(raw, units=units).df["ZONE"].to_numpy(float)
        assert set(np.unique(loaded[np.isfinite(loaded)])) == {1.0, 2.0, 3.0}


class TestZoneOfLobe:
    """A top inside an event's lobe is a top that event is carrying.

    Comparing only the two samples at the extremum was right when every
    interface was a candidate reflector.  With events picked off the trace
    there are far fewer of them, and a top almost never lands exactly between
    one event's two samples, so that flag went quiet.
    """

    @staticmethod
    def _bounds(upper, lower, resolved=True):
        return {"upper_start": np.array([upper[0]]),
                "upper_stop": np.array([upper[1]]),
                "lower_start": np.array([lower[0]]),
                "lower_stop": np.array([lower[1]]),
                "resolved": np.array([resolved])}

    def test_a_top_inside_the_lobe_is_flagged(self):
        labels = np.array(["Upper"] * 20 + ["Reservoir"] * 20, dtype=object)
        found = zone_of_lobe(labels, self._bounds((12, 20), (19, 28)))
        assert found["zone"][0] == "Upper"
        assert found["zone_below"][0] == "Reservoir"
        assert bool(found["is_zone_boundary"][0])

    def test_an_event_well_inside_one_zone_is_not_flagged(self):
        labels = np.array(["Upper"] * 20 + ["Reservoir"] * 20, dtype=object)
        found = zone_of_lobe(labels, self._bounds((2, 9), (8, 15)))
        assert found["zone"][0] == found["zone_below"][0] == "Upper"
        assert not bool(found["is_zone_boundary"][0])

    def test_the_interface_flag_misses_what_the_lobe_catches(self):
        """The reason this function exists, stated as a comparison."""
        from avo_qi.core.zones import zone_of_interface

        labels = np.array(["Upper"] * 20 + ["Reservoir"] * 20, dtype=object)
        # The event sits at sample 16, four samples above the top at 20, so
        # samples 16 and 17 are both in "Upper" and the interface test is blind
        # to a boundary its lobe plainly spans.
        assert not zone_of_interface(labels, [16])["is_zone_boundary"][0]
        assert zone_of_lobe(labels, self._bounds((12, 17), (16, 26)),
                            samples=[16])["is_zone_boundary"][0]

    def test_an_unresolved_lobe_falls_back_to_the_interface(self):
        labels = np.array(["Upper"] * 20 + ["Reservoir"] * 20, dtype=object)
        found = zone_of_lobe(labels, self._bounds((0, 0), (0, 0), resolved=False),
                             samples=[19])
        assert found["zone"][0] == "Upper"
        assert found["zone_below"][0] == "Reservoir"

    def test_no_labels_at_all_is_not_an_error(self):
        found = zone_of_lobe(np.array([], dtype=object), self._bounds((0, 1), (1, 2)))
        assert found["zone"].size == 0
