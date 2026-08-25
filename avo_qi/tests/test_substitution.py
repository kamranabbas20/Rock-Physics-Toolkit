"""Tests for attaching computed fluid cases to a loaded well."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from avo_qi.io.loader import (
    add_standard_cases,
    add_substituted_case,
    case_column,
    detect_fluid_cases,
    standardise,
)
from avo_qi.core.rockphysics import FLUIDS, MINERALS

K_QUARTZ = MINERALS["quartz"][0]
BRINE, GAS = FLUIDS["brine"], FLUIDS["gas"]


@pytest.fixture()
def well():
    """A short brine-filled sand, built from values that do substitute."""
    n = 12
    return standardise(pd.DataFrame({
        "DEPT": np.linspace(2000.0, 2011.0, n),
        "VP": np.full(n, 2830.2),
        "VS": np.full(n, 1498.9),
        "RHOB": np.full(n, 2.2444),
        "PHI": np.full(n, 0.26),
    }), name="test well")


class TestAddCase:
    def test_the_new_case_appears_and_is_marked_computed(self, well):
        add_substituted_case(well, "gas", BRINE, GAS, 0.26, K_QUARTZ)
        assert "gas" in well.cases
        assert well.is_computed("gas")
        assert not well.is_computed("in situ")
        assert well.has_fluid_cases

    def test_the_curves_land_where_every_other_page_looks_for_them(self, well):
        """The whole point of the naming: nothing downstream needs changing."""
        add_substituted_case(well, "gas", BRINE, GAS, 0.26, K_QUARTZ)
        for curve in ("VP", "VS", "RHOB"):
            assert case_column(curve, "gas") in well.df.columns
        assert "gas" in detect_fluid_cases(well.df.columns)

    def test_frame_swaps_to_the_computed_case(self, well):
        add_substituted_case(well, "gas", BRINE, GAS, 0.26, K_QUARTZ)
        original = well.frame("in situ")["VP"].to_numpy()
        gassy = well.frame("gas")["VP"].to_numpy()
        assert np.all(gassy < original)
        assert float(gassy[0]) == pytest.approx(2315.8, abs=0.1)

    def test_the_source_case_is_left_alone(self, well):
        before = well.frame("in situ")["VP"].to_numpy().copy()
        add_substituted_case(well, "gas", BRINE, GAS, 0.26, K_QUARTZ)
        assert np.allclose(well.frame("in situ")["VP"].to_numpy(), before)

    def test_a_note_records_what_was_done(self, well):
        add_substituted_case(well, "gas", BRINE, GAS, 0.26, K_QUARTZ)
        assert any("Gassmann-substituted here" in n for n in well.notes)

    def test_the_note_counts_the_samples_that_could_not_be_substituted(self, well):
        phi = np.full(len(well.df), 0.26)
        phi[:3] = 1.5                       # impossible porosity
        result = add_substituted_case(well, "gas", BRINE, GAS, phi, K_QUARTZ)
        assert int((~result["valid"]).sum()) == 3
        assert any("3 of 12 samples could not be substituted" in n for n in well.notes)
        assert np.isnan(well.frame("gas")["VP"].to_numpy()[:3]).all()

    def test_cases_come_back_in_the_conventional_order(self, well):
        add_substituted_case(well, "gas", BRINE, GAS, 0.26, K_QUARTZ)
        add_substituted_case(well, "oil", BRINE, FLUIDS["oil"], 0.26, K_QUARTZ)
        assert well.cases == ["in situ", "oil", "gas"]

    def test_substituting_twice_replaces_rather_than_duplicates(self, well):
        add_substituted_case(well, "gas", BRINE, GAS, 0.26, K_QUARTZ)
        first = well.frame("gas")["VP"].to_numpy().copy()
        add_substituted_case(well, "gas", BRINE, (0.05, 0.30), 0.26, K_QUARTZ)
        assert well.cases.count("gas") == 1
        assert not np.allclose(well.frame("gas")["VP"].to_numpy(), first)

    def test_can_substitute_from_a_case_other_than_the_active_one(self, well):
        add_substituted_case(well, "gas", BRINE, GAS, 0.26, K_QUARTZ)
        add_substituted_case(well, "oil", GAS, FLUIDS["oil"], 0.26, K_QUARTZ,
                             source_case="gas")
        # Round tripping brine -> gas -> oil must land on brine -> oil.
        direct = well.frame("oil")["VP"].to_numpy()
        assert float(direct[0]) == pytest.approx(2496.8, abs=0.2)

    def test_accepts_fluid_properties_that_vary_down_the_well(self, well):
        n = len(well.df)
        k_gas = np.linspace(0.02, 0.09, n)
        result = add_substituted_case(well, "gas", BRINE, (k_gas, 0.25), 0.26,
                                      K_QUARTZ)
        vp = np.asarray(result["VP"])
        assert np.all(np.diff(vp) > 0)      # stiffer gas, faster rock

    def test_rejects_a_case_the_toolkit_does_not_know(self, well):
        with pytest.raises(ValueError, match="is not one of"):
            well.add_case("supercritical CO2", [1.0], [1.0], [1.0])

    def test_rejects_curves_of_the_wrong_length(self, well):
        with pytest.raises(ValueError, match="samples but the well has"):
            well.add_case("gas", [1.0, 2.0], [1.0, 2.0], [1.0, 2.0])

    def test_a_case_can_be_declared_loaded_rather_than_computed(self, well):
        n = len(well.df)
        well.add_case("gas", np.full(n, 2300.0), np.full(n, 1570.0),
                      np.full(n, 2.03), computed=False)
        assert "gas" in well.cases
        assert not well.is_computed("gas")


class TestTheStandardSuite:
    """A well that arrives with one set of logs and nothing else still has to
    be comparable across fluids, which means modelling the cases here."""

    @staticmethod
    def _well():
        """The demo well with its own fluid cases stripped off, which is what a
        well carrying only in-situ logs looks like."""
        import os

        from avo_qi.io.loader import read_well

        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        frame, units = read_well(os.path.join(here, "sample_data", "demo_well.las"))
        # Drop by detection rather than by guessing suffixes: this file spells
        # its brine case VP_BR, not VP_BRINE.
        substituted = {column
                       for case, curves in detect_fluid_cases(frame.columns).items()
                       if case != "in situ" for column in curves.values()}
        keep = [c for c in frame.columns if c not in substituted]
        return standardise(frame[keep], units=units, name="unsubstituted")

    def test_all_three_cases_appear_and_are_marked_computed(self):
        well = self._well()
        assert well.cases == ["in situ"]
        add_standard_cases(well, well.df["PHI"].to_numpy(float), k_mineral=K_QUARTZ)
        assert well.cases == ["in situ", "brine", "oil", "gas"]
        assert well.computed_cases == ["brine", "oil", "gas"]
        assert not well.is_computed("in situ")

    def test_the_wells_own_logs_stay_reachable(self):
        """Adding a model must never make the measurement unreachable."""
        well = self._well()
        before = well.df["VP"].to_numpy(float).copy()
        add_standard_cases(well, well.df["PHI"].to_numpy(float), k_mineral=K_QUARTZ)
        assert well.frame("in situ")["VP"].to_numpy(float) == pytest.approx(before)
        assert well.df["VP"].to_numpy(float) == pytest.approx(before)

    def test_the_cases_come_out_in_their_physical_order(self):
        """Brine is the stiffest and heaviest pore fluid, gas the softest and
        lightest, and the rock has to follow — in the reservoir, where the
        substitution actually ran."""
        well = self._well()
        phi = well.df["PHI"].to_numpy(float)
        vsh = well.df["VSH"].to_numpy(float)
        results = add_standard_cases(well, phi, k_mineral=K_QUARTZ,
                                     reservoir=vsh <= 0.35)
        # Only where the substitution actually ran: elsewhere every case keeps
        # the in-situ curves, so all three are equal by construction.
        ran = results["brine"]["valid"]
        assert ran.sum() > 100

        rho = {c: well.frame(c)["RHOB"].to_numpy(float)[ran]
               for c in ("brine", "oil", "gas")}
        assert (rho["brine"] > rho["oil"]).all() and (rho["oil"] > rho["gas"]).all()

        # Velocity in the porous sand, where the fluid can actually move the
        # frame; see the stiff-rock exception below.
        soft = ran & (phi > 0.20)
        assert soft.sum() > 50
        vp = {c: well.frame(c)["VP"].to_numpy(float)[soft]
              for c in ("brine", "oil", "gas")}
        assert (vp["brine"] > vp["oil"]).all() and (vp["oil"] > vp["gas"]).all()

    def test_in_a_stiff_rock_a_lighter_fluid_can_raise_the_velocity(self):
        """Vp = sqrt((K + 4/3 mu)/rho). In a tight, stiff frame the pore fluid
        barely moves K but still moves rho, so gas can come out marginally
        *faster* than oil. "Brine, then oil, then gas" is a soft-rock result,
        not a universal one, and a tool that enforced it would be wrong."""
        well = self._well()
        phi = well.df["PHI"].to_numpy(float)
        results = add_standard_cases(well, phi, k_mineral=K_QUARTZ,
                                     reservoir=well.df["VSH"].to_numpy(float) <= 0.35)
        stiff = results["oil"]["valid"] & (phi < 0.20)
        assert stiff.sum() > 10, "the demo well's cemented streak should be here"
        oil = well.frame("oil")["VP"].to_numpy(float)[stiff]
        gas = well.frame("gas")["VP"].to_numpy(float)[stiff]
        # The ordering simply stops holding: gas comes out faster at most of
        # these samples and slower at the rest, all of it inside a fraction of
        # a percent, which is the honest description of a fluid effect that has
        # run out of frame to act on.
        assert (gas > oil).mean() > 0.5
        assert (np.abs(gas - oil) / oil < 0.01).all()

        # In the porous sand the same comparison is unambiguous.
        porous = results["oil"]["valid"] & (phi > 0.20)
        assert (well.frame("gas")["VP"].to_numpy(float)[porous]
                < well.frame("oil")["VP"].to_numpy(float)[porous]).all()

    def test_substituting_back_into_the_fluid_that_is_there_changes_nothing(self):
        """The consistency check the whole suite rests on: the demo well's sand
        holds gas, so modelling its gas case must return the logs themselves."""
        well = self._well()
        depth = well.df["DEPTH"].to_numpy(float)
        sand = (depth >= 2040) & (depth < 2070)
        add_standard_cases(well, well.df["PHI"].to_numpy(float), k_mineral=K_QUARTZ,
                           in_situ_hydrocarbon="gas",
                           sw=well.df["SW"].to_numpy(float), hydrocarbon_sw=0.2,
                           reservoir=sand)
        original = well.frame("in situ")["VP"].to_numpy(float)[sand]
        modelled = well.frame("gas")["VP"].to_numpy(float)[sand]
        assert modelled == pytest.approx(original, rel=0.02)
        assert not np.array_equal(modelled, original)   # it did run, not skip

    def test_naming_the_wrong_in_situ_fluid_fails_loudly(self):
        """Why the fluid already in the pores is asked for. The demo well's
        sand holds gas; call it brine-filled and Gassmann is being told the
        rock is stiffer than the logs say, so the dry frame comes out negative
        and *every* sample is refused. It does not quietly put the gas effect
        in twice — which is the failure mode worth having."""
        outcome = {}
        for hydrocarbon in (None, "gas"):
            well = self._well()
            depth = well.df["DEPTH"].to_numpy(float)
            sand = (depth >= 2040) & (depth < 2070)
            outcome[hydrocarbon] = add_standard_cases(
                well, well.df["PHI"].to_numpy(float), k_mineral=K_QUARTZ,
                in_situ_hydrocarbon=hydrocarbon,
                sw=well.df["SW"].to_numpy(float), reservoir=sand)["gas"]

        assert outcome["gas"]["valid"].sum() > 100
        assert outcome[None]["valid"].sum() == 0
        reasons = {str(r) for r in outcome[None]["reasons"][~outcome[None]["valid"]]}
        assert "dry-frame modulus is negative" in reasons

    def test_where_gassmann_cannot_run_the_original_logs_are_kept(self):
        """A blank shale above a substituted sand deletes the very interface the
        case was built to show, so the seal keeps its own curves."""
        well = self._well()
        vsh = well.df["VSH"].to_numpy(float)
        shale = vsh > 0.35
        add_standard_cases(well, well.df["PHI"].to_numpy(float), k_mineral=K_QUARTZ,
                           reservoir=~shale)
        for case in ("brine", "oil", "gas"):
            frame = well.frame(case)
            assert frame["VP"].notna().all(), case
            assert frame["VP"].to_numpy(float)[shale] == pytest.approx(
                well.frame("in situ")["VP"].to_numpy(float)[shale])

    def test_the_note_says_how_many_samples_kept_their_own_values(self):
        well = self._well()
        add_standard_cases(well, well.df["PHI"].to_numpy(float), k_mineral=K_QUARTZ)
        joined = " ".join(well.notes)
        assert "Gassmann-substituted here" in joined
        assert "kept their in situ values" in joined

    def test_a_case_it_cannot_model_is_refused_rather_than_faked(self):
        well = self._well()
        with pytest.raises(ValueError, match="cannot model"):
            add_standard_cases(well, well.df["PHI"].to_numpy(float),
                               cases=("in situ",))
