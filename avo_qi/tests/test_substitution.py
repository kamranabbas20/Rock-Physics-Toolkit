"""Tests for attaching computed fluid cases to a loaded well."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from avo_qi.io.loader import (
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
