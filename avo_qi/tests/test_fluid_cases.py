"""Fluid-case handling: detection, standardisation and cross-case comparison.

Substitution itself is done upstream — these tests cover recognising the
substituted curves (``VP_BR``, ``VS_OIL``, ``RHOB_GAS`` ...) and comparing the
resulting AVO responses.  Nothing here performs a substitution.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from avo_qi.core.avo import compare_cases, reflector_avo
from avo_qi.core.reflectivity import reflectivity_series
from avo_qi.io.loader import (
    _split_fluid_suffix,
    case_column,
    depth_to_twt,
    detect_fluid_cases,
    read_well,
    resample_to_time,
    standardise,
)

DEMO = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "sample_data", "demo_well.las",
)


class TestSuffixSplitting:
    @pytest.mark.parametrize(
        "mnemonic,expected",
        [
            ("VP", ("VP", None)),
            ("VS", ("VS", None)),
            ("RHOB", ("RHOB", None)),
            ("VP_BR", ("VP", "brine")),
            ("VP_BRINE", ("VP", "brine")),
            ("vs_oil", ("VS", "oil")),
            ("RHOB_GAS", ("RHOB", "gas")),
            ("VPGAS", ("VP", "gas")),
            ("VP_INSITU", ("VP", "in situ")),
            ("DTC_BR", ("VP", "brine")),
        ],
    )
    def test_recognised_forms(self, mnemonic, expected):
        assert _split_fluid_suffix(mnemonic) == expected

    @pytest.mark.parametrize("mnemonic", ["VSH", "VSHALE", "PHI", "SW", "GR", "DEPT"])
    def test_non_case_curves_are_rejected(self, mnemonic):
        """VSH must never be read as a shear log for a fluid called 'H'."""
        assert _split_fluid_suffix(mnemonic) == (None, None)

    def test_vshale_survives_alongside_a_real_shear_case(self):
        cases = detect_fluid_cases(["VP", "VS", "RHOB", "VSH", "VP_BR", "VS_BR", "RHOB_BR"])
        assert set(cases) == {"in situ", "brine"}
        assert "VSH" not in cases["in situ"].values()
        assert cases["in situ"]["VS"] == "VS"


class TestCaseDetection:
    COLUMNS = [
        "DEPT", "VP", "VS", "RHOB",
        "VP_BR", "VS_BR", "RHOB_BR",
        "VP_OIL", "VS_OIL", "RHOB_OIL",
        "VP_GAS", "VS_GAS", "RHOB_GAS",
        "GR", "VSH", "PHI", "SW",
    ]

    def test_finds_every_case(self):
        cases = detect_fluid_cases(self.COLUMNS)
        assert list(cases) == ["in situ", "brine", "oil", "gas"]
        assert cases["gas"] == {"VP": "VP_GAS", "VS": "VS_GAS", "RHOB": "RHOB_GAS"}

    def test_incomplete_cases_are_dropped(self):
        # An oil case with no density is not a case.
        cases = detect_fluid_cases(["VP", "VS", "RHOB", "VP_OIL", "VS_OIL"])
        assert list(cases) == ["in situ"]

    def test_cases_come_back_in_canonical_order(self):
        shuffled = ["VP_GAS", "VS_GAS", "RHOB_GAS", "VP_BR", "VS_BR", "RHOB_BR"]
        assert list(detect_fluid_cases(shuffled)) == ["brine", "gas"]

    def test_no_cases_when_nothing_matches(self):
        assert detect_fluid_cases(["DEPT", "GR", "PHI"]) == {}


@pytest.fixture(scope="module")
def well():
    raw, units = read_well(DEMO)
    return standardise(raw, units=units, name="DEMO-1")


class TestStandardiseWithCases:
    def test_demo_well_carries_four_cases(self, well):
        assert well.cases == ["in situ", "brine", "oil", "gas"]
        assert well.has_fluid_cases
        assert well.active_case == "in situ"

    def test_every_case_gets_its_own_columns(self, well):
        for case in well.cases:
            for curve in ("VP", "VS", "RHOB"):
                assert case_column(curve, case) in well.df.columns

    def test_active_case_occupies_the_plain_columns(self, well):
        vp, vs, rho = well.logs()
        assert np.allclose(vp, well.df[case_column("VP", "in situ")])
        assert np.allclose(rho, well.df[case_column("RHOB", "in situ")])

    def test_frame_swaps_the_active_curves(self, well):
        gas = well.frame("gas")
        assert np.allclose(gas["VP"], well.df["VP_GAS"])
        assert np.allclose(gas["RHOB"], well.df["RHOB_GAS"])
        # The shared curves are untouched by the swap.
        assert np.allclose(gas["GR"], well.df["GR"], equal_nan=True)

    def test_gas_case_is_softer_and_lighter_than_the_brine_case(self, well):
        depth = well.depth
        sand = (depth >= 2045) & (depth < 2065)
        vp_b, vs_b, rho_b = (a[sand] for a in well.logs("brine"))
        vp_g, vs_g, rho_g = (a[sand] for a in well.logs("gas"))
        assert vp_g.mean() < vp_b.mean()          # gas slows the P wave
        assert rho_g.mean() < rho_b.mean()        # and lightens the rock
        assert vs_g.mean() > vs_b.mean()          # shear speeds up as density drops
        assert (vp_g / vs_g).mean() < (vp_b / vs_b).mean() - 0.4

    def test_shale_is_the_same_in_every_case(self, well):
        depth = well.depth
        shale = (depth >= 2005) & (depth < 2035)
        reference = well.logs("brine")[0][shale]
        for case in well.cases:
            assert np.allclose(well.logs(case)[0][shale], reference)

    def test_selecting_an_explicit_case_on_load(self):
        raw, units = read_well(DEMO)
        well = standardise(raw, units=units, case="gas")
        assert well.active_case == "gas"
        assert np.allclose(well.logs()[0], well.df["VP_GAS"])

    def test_unknown_case_is_rejected(self):
        raw, units = read_well(DEMO)
        with pytest.raises(ValueError):
            standardise(raw, units=units, case="condensate")
        well = standardise(raw, units=units)
        with pytest.raises(ValueError):
            well.frame("condensate")

    def test_load_note_reports_the_cases(self, well):
        assert any("fluid cases found" in note for note in well.notes)


def _case_tables(well, angles, t0=1.6, dt=0.001, threshold=0.03, reference="brine"):
    """Fit every fluid case at the same interfaces on one shared time axis."""
    ref_frame = well.complete(well.active_case).reset_index(drop=True)
    twt = depth_to_twt(ref_frame["DEPTH"].to_numpy(float),
                       ref_frame["VP"].to_numpy(float), t0=t0)

    frames, rcs = {}, {}
    for case in well.cases:
        frame = resample_to_time(well.complete(case).reset_index(drop=True), twt, dt=dt)
        frames[case] = frame
        rcs[case] = reflectivity_series(
            *(frame[k].to_numpy(float) for k in ("VP", "VS", "RHOB")), angles
        )

    base = reflector_avo(rcs[reference], angles=angles, threshold=threshold)
    samples = base["sample"].to_numpy()
    tables = {
        case: reflector_avo(
            rcs[case], angles=angles, samples=samples,
            depth=frames[case]["DEPTH"].to_numpy(float),
            twt=frames[case]["TWT"].to_numpy(float),
        )
        for case in well.cases
    }
    return tables, frames


class TestSharedTimeAxis:
    """Each case must land on the same time grid, or reflectors misalign."""

    def test_every_case_lands_on_the_same_grid(self, well):
        _, frames = _case_tables(well, np.arange(0.0, 41.0, 2.0))
        reference = frames[well.active_case]["TWT"].to_numpy()
        for case, frame in frames.items():
            assert frame["TWT"].to_numpy().shape == reference.shape
            assert np.allclose(frame["TWT"].to_numpy(), reference)

    def test_each_case_keeps_its_own_depth_at_a_given_sample(self, well):
        _, frames = _case_tables(well, np.arange(0.0, 41.0, 2.0))
        depths = [f["DEPTH"].to_numpy() for f in frames.values()]
        for other in depths[1:]:
            assert np.allclose(depths[0], other)

    def test_a_per_case_time_axis_would_misalign_the_sands(self, well):
        """Guards the bug this design avoids: integrating each case's own Vp
        shifts the deeper reflectors between cases."""
        brine = well.complete("brine").reset_index(drop=True)
        gas = well.complete("gas").reset_index(drop=True)
        twt_brine = depth_to_twt(brine["DEPTH"].to_numpy(float),
                                 brine["VP"].to_numpy(float), t0=1.6)
        twt_gas = depth_to_twt(gas["DEPTH"].to_numpy(float),
                               gas["VP"].to_numpy(float), t0=1.6)
        # Identical above the first sand, materially different below it.
        assert twt_brine[10] == pytest.approx(twt_gas[10], abs=1e-9)
        assert abs(twt_brine[-1] - twt_gas[-1]) > 2e-3


ANGLES = np.arange(0.0, 41.0, 2.0)


@pytest.fixture(scope="module")
def comparison(well):
    tables, _ = _case_tables(well, ANGLES)
    return compare_cases(tables, reference="brine")


class TestCompareCases:
    ANGLES = ANGLES

    def test_shape_and_columns(self, comparison):
        assert len(comparison) > 0
        for case in ("in situ", "brine", "oil", "gas"):
            for prefix in ("A_", "B_", "class_"):
                assert f"{prefix}{case}" in comparison.columns
        assert "fluid_vector" in comparison.columns
        assert "class_changed" in comparison.columns

    def test_reference_has_no_fluid_vector_columns(self, comparison):
        assert "dA_brine" not in comparison.columns
        assert "dA_gas" in comparison.columns

    def test_gas_sand_top_turns_class_three_only_with_gas(self, comparison):
        top = comparison.iloc[(comparison["depth"] - 2039.6).abs().argmin()]
        assert top["class_gas"] == "III"
        assert top["A_gas"] < top["A_brine"]        # gas darkens the intercept
        assert top["class_brine"] != "III"
        assert bool(top["class_changed"]) is True

    def test_the_stiff_streak_barely_moves_with_fluid(self, comparison):
        """A cemented frame is fluid-insensitive; that is why it is in the well."""
        streak = comparison.iloc[(comparison["depth"] - 2119.1).abs().argmin()]
        gas_sand = comparison.iloc[(comparison["depth"] - 2039.6).abs().argmin()]
        assert abs(streak["A_gas"] - streak["A_brine"]) < 0.05
        assert abs(gas_sand["A_gas"] - gas_sand["A_brine"]) > 0.1

    def test_fluid_vector_is_the_longest_case_displacement(self, comparison):
        row = comparison.iloc[0]
        lengths = [
            np.hypot(row[f"dA_{c}"], row[f"dB_{c}"]) for c in ("in situ", "oil", "gas")
        ]
        assert row["fluid_vector"] == pytest.approx(max(lengths))

    def test_reference_case_is_recorded(self, comparison):
        assert comparison.attrs["reference_case"] == "brine"

    def test_defaults_to_brine_as_reference(self, well):
        tables, _ = _case_tables(well, self.ANGLES)
        assert compare_cases(tables).attrs["reference_case"] == "brine"

    def test_rejects_an_unknown_reference(self):
        frame = pd.DataFrame({"sample": [1], "A_shuey": [0.1], "B_shuey": [-0.2],
                              "avo_class": ["I"]})
        with pytest.raises(ValueError):
            compare_cases({"brine": frame}, reference="gas")
        with pytest.raises(ValueError):
            compare_cases({})

    def test_single_case_has_a_zero_fluid_vector(self):
        frame = pd.DataFrame({"sample": [1, 2], "A_shuey": [0.1, -0.2],
                              "B_shuey": [-0.2, 0.3], "avo_class": ["I", "IV"]})
        merged = compare_cases({"brine": frame})
        assert np.all(merged["fluid_vector"] == 0.0)
        assert not merged["class_changed"].any()


class TestForcedReflectorSamples:
    def test_samples_override_the_threshold(self):
        rc = np.zeros((50, 5))
        rc[10, :] = -0.2
        angles = np.arange(0.0, 41.0, 10.0)
        forced = reflector_avo(rc, angles=angles, samples=[3, 10, 20])
        assert forced["sample"].tolist() == [3, 10, 20]
        # Sample 3 is a flat zero interface, kept because it was asked for.
        assert forced.loc[forced["sample"] == 3, "A_shuey"].iloc[0] == pytest.approx(0.0)

    def test_out_of_range_samples_are_rejected(self):
        rc = np.zeros((10, 3))
        with pytest.raises(ValueError):
            reflector_avo(rc, angles=np.arange(0.0, 31.0, 10.0), samples=[99])
