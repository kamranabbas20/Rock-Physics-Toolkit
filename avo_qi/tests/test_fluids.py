"""Tests for Batzle-Wang pore-fluid properties.

The correlations are empirical, so most of these check the physics they have
to obey — the ideal-gas limit, monotonicity in pressure and salinity, live oil
being softer than dead — rather than pinning digits.  Where a published number
does exist and is unambiguous, such as the speed of sound in pure water at room
conditions, it is used directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from avo_qi.core.fluids import (
    BRINE_PRESETS,
    CONDITION_DEFAULTS,
    FITTED_RANGE,
    FLUID_DEFAULTS,
    GAS_PRESETS,
    OIL_PRESETS,
    brine_properties,
    fluid_suite,
    mix_two_fluids,
    fluid_properties,
    gas_properties,
    geothermal_temperature,
    hydrostatic_pressure,
    in_range,
    oil_properties,
)


def _velocity(k, rho):
    """Back out velocity in m/s from K in GPa and rho in g/cc."""
    return float(np.sqrt(k / rho * 1e6))


class TestBrine:
    def test_pure_water_matches_the_measured_speed_of_sound(self):
        """1496.7 m/s at 25 C and atmospheric pressure, a well-measured value."""
        k, rho = brine_properties(0.1, 25.0, salinity=0.0)
        assert _velocity(k, rho) == pytest.approx(1496.7, abs=1.0)
        assert float(rho) == pytest.approx(0.997, abs=0.002)

    def test_seawater_is_faster_and_denser_than_pure_water(self):
        k_fresh, rho_fresh = brine_properties(0.1, 25.0, salinity=0.0)
        k_sea, rho_sea = brine_properties(0.1, 25.0, salinity=0.035)
        assert _velocity(k_sea, rho_sea) == pytest.approx(1531.0, abs=5.0)
        assert float(rho_sea) > float(rho_fresh)
        assert float(k_sea) > float(k_fresh)

    def test_stiffens_with_pressure(self):
        k = [float(brine_properties(p, 90.0, 0.05)[0]) for p in (10.0, 30.0, 60.0)]
        assert k[0] < k[1] < k[2]

    def test_stiffens_with_salinity(self):
        k = [float(brine_properties(30.0, 90.0, s)[0]) for s in (0.0, 0.05, 0.15)]
        assert k[0] < k[1] < k[2]

    def test_reservoir_brine_is_stiffer_than_the_fixed_table_entry(self):
        """The table's 2.80 GPa is a room-condition figure; at depth it is more."""
        k, rho = brine_properties(30.0, 90.0, 0.05)
        assert 2.5 < float(k) < 3.5
        assert 0.98 < float(rho) < 1.10


class TestGas:
    def test_approaches_the_ideal_gas_limit_at_low_pressure(self):
        """For an ideal gas K = gamma * P, and gamma is about 1.32 for methane.

        This is an independent physical constraint rather than another
        empirical number, so it is the strongest check available here.
        """
        for p in (0.5, 1.0):
            k, _ = gas_properties(p, 25.0, gravity=0.6)
            assert float(k) * 1000.0 / p == pytest.approx(1.32, abs=0.12)

    def test_reservoir_gas_is_far_stiffer_and_denser_than_the_table(self):
        """The fixed table entry (0.021 GPa, 0.25 g/cc) is not self-consistent.

        A gas dense enough to be 0.25 g/cc is at high pressure, where its
        modulus is nearer 0.1 GPa than 0.021 — which is the reason this module
        exists.
        """
        k, rho = gas_properties(30.0, 90.0, gravity=0.6)
        assert 0.04 < float(k) < 0.12
        assert 0.13 < float(rho) < 0.25

    def test_denser_and_stiffer_with_pressure(self):
        pressures = np.array([10.0, 20.0, 30.0, 45.0])
        k, rho = gas_properties(pressures, 90.0)
        assert np.all(np.diff(k) > 0)
        assert np.all(np.diff(rho) > 0)

    def test_lighter_and_softer_when_hotter(self):
        k_cool, rho_cool = gas_properties(30.0, 60.0)
        k_hot, rho_hot = gas_properties(30.0, 140.0)
        assert float(rho_hot) < float(rho_cool)
        assert float(k_hot) < float(k_cool)

    def test_a_heavier_gas_is_denser(self):
        _, light = gas_properties(30.0, 90.0, gravity=0.56)
        _, heavy = gas_properties(30.0, 90.0, gravity=0.85)
        assert float(heavy) > float(light)

    def test_is_always_much_softer_than_brine(self):
        k_gas, _ = gas_properties(30.0, 90.0)
        k_brine, _ = brine_properties(30.0, 90.0, 0.05)
        assert float(k_gas) < 0.1 * float(k_brine)


class TestOil:
    def test_dead_oil_lands_in_the_expected_range(self):
        k, rho = oil_properties(30.0, 90.0, api=30.0, gor=0.0)
        assert 0.7 < float(k) < 2.0
        assert 0.75 < float(rho) < 0.92

    def test_dissolved_gas_softens_and_lightens_it(self):
        gors = [0.0, 50.0, 100.0, 200.0]
        k = [float(oil_properties(30.0, 90.0, 30.0, g)[0]) for g in gors]
        rho = [float(oil_properties(30.0, 90.0, 30.0, g)[1]) for g in gors]
        assert all(a > b for a, b in zip(k, k[1:]))
        assert all(a > b for a, b in zip(rho, rho[1:]))
        # A GOR of 200 L/L should roughly halve the modulus, not nudge it.
        assert k[-1] < 0.6 * k[0]

    def test_a_lighter_oil_is_less_dense_and_softer(self):
        k_heavy, rho_heavy = oil_properties(30.0, 90.0, api=15.0)
        k_light, rho_light = oil_properties(30.0, 90.0, api=45.0)
        assert float(rho_light) < float(rho_heavy)
        assert float(k_light) < float(k_heavy)

    def test_stiffens_with_pressure_and_softens_with_temperature(self):
        assert float(oil_properties(50.0, 90.0)[0]) > float(oil_properties(15.0, 90.0)[0])
        assert float(oil_properties(30.0, 140.0)[0]) < float(oil_properties(30.0, 40.0)[0])

    def test_sits_between_gas_and_brine(self):
        k_oil, _ = oil_properties(30.0, 90.0, api=30.0, gor=50.0)
        k_gas, _ = gas_properties(30.0, 90.0)
        k_brine, _ = brine_properties(30.0, 90.0, 0.05)
        assert float(k_gas) < float(k_oil) < float(k_brine)


class TestDepthHelpers:
    def test_hydrostatic_pressure_is_ten_megapascals_per_kilometre(self):
        assert float(hydrostatic_pressure(2000.0)) == pytest.approx(20.0)
        assert float(hydrostatic_pressure(2000.0, gradient=12.0)) == pytest.approx(24.0)

    def test_temperature_follows_the_geothermal_gradient(self):
        assert float(geothermal_temperature(2000.0)) == pytest.approx(75.0)
        assert float(geothermal_temperature(0.0)) == pytest.approx(15.0)

    def test_a_typical_reservoir_depth_lands_inside_the_fitted_range(self):
        depth = 2500.0
        assert bool(in_range(hydrostatic_pressure(depth),
                             geothermal_temperature(depth)))


class TestFacade:
    @pytest.mark.parametrize("fluid", ["brine", "oil", "gas"])
    def test_agrees_with_the_individual_routines(self, fluid):
        k, rho, _ = fluid_properties(fluid, 30.0, 90.0)
        direct = {"brine": lambda: brine_properties(30.0, 90.0, 0.035),
                  "oil": lambda: oil_properties(30.0, 90.0, 30.0, 0.0, 0.6),
                  "gas": lambda: gas_properties(30.0, 90.0, 0.6)}[fluid]()
        assert k == pytest.approx(float(direct[0]))
        assert rho == pytest.approx(float(direct[1]))

    def test_is_quiet_inside_the_fitted_range(self):
        _, _, warning = fluid_properties("brine", 30.0, 90.0)
        assert warning is None

    def test_says_so_when_extrapolating(self):
        p_lo = FITTED_RANGE["pressure"][0]
        _, _, warning = fluid_properties("brine", p_lo / 2.0, 90.0)
        assert warning is not None
        assert "outside" in warning and "extrapolation" in warning

    def test_counts_how_many_samples_are_out_of_range(self):
        pressure = np.array([1.0, 30.0, 40.0])
        _, _, warning = fluid_properties("brine", pressure, 90.0)
        assert warning.startswith("1 of 3")

    def test_rejects_an_unknown_fluid(self):
        with pytest.raises(ValueError, match="unknown fluid"):
            fluid_properties("mercury", 30.0, 90.0)

    def test_broadcasts_over_a_depth_log(self):
        depth = np.linspace(1500.0, 3500.0, 40)
        k, rho, _ = fluid_properties("gas", hydrostatic_pressure(depth),
                                     geothermal_temperature(depth))
        assert k.shape == depth.shape == rho.shape
        assert np.all(np.isfinite(k)) and np.all(k > 0)


class TestIndustryDefaults:
    """A well with no fluid cases still has to be substitutable, which means
    starting values for fluids nobody has measured here."""

    def test_the_defaults_are_ordinary_industry_values(self):
        assert FLUID_DEFAULTS["salinity"] == pytest.approx(0.035)   # seawater
        assert 20.0 < FLUID_DEFAULTS["api"] < 45.0
        assert 0.5 < FLUID_DEFAULTS["gas_gravity"] < 1.0
        assert CONDITION_DEFAULTS["pressure_gradient"] == pytest.approx(10.0)

    def test_every_preset_is_a_usable_argument(self):
        """A preset that does not evaluate is a trap, not a convenience."""
        for salinity in BRINE_PRESETS.values():
            k, rho = brine_properties(30.0, 90.0, salinity)
            assert 1.0 < k < 5.0 and 0.9 < rho < 1.25
        for api, gor in OIL_PRESETS.values():
            k, rho = oil_properties(30.0, 90.0, api, gor)
            assert 0.05 < k < 2.5 and 0.4 < rho < 1.05
        for gravity in GAS_PRESETS.values():
            k, rho = gas_properties(30.0, 90.0, gravity)
            assert 0.0 < k < 0.5 and 0.05 < rho < 0.5

    def test_the_presets_are_ordered_the_way_they_are_named(self):
        saltier = list(BRINE_PRESETS.values())
        assert saltier == sorted(saltier)
        heavier = [k for k, _ in OIL_PRESETS.values()]
        assert heavier == sorted(heavier, reverse=True)   # API falls as oil thickens


class TestFluidSuite:
    DEPTH = np.array([1000.0, 2000.0, 3000.0, 4000.0])

    def test_it_puts_every_fluid_on_the_wells_own_conditions(self):
        suite = fluid_suite(self.DEPTH)
        assert suite["pressure"] == pytest.approx([10.0, 20.0, 30.0, 40.0])
        assert suite["temperature"] == pytest.approx([45.0, 75.0, 105.0, 135.0])
        for case in ("brine", "oil", "gas"):
            k, rho = suite[case]
            assert k.shape == self.DEPTH.shape
            assert np.isfinite(k).all() and np.isfinite(rho).all()

    def test_the_fluids_stay_in_their_physical_order(self):
        """Brine is the stiffest and densest, gas the softest and lightest —
        at every depth, or a substitution would move the rock the wrong way."""
        suite = fluid_suite(self.DEPTH)
        k_brine, rho_brine = suite["brine"]
        k_oil, rho_oil = suite["oil"]
        k_gas, rho_gas = suite["gas"]
        assert (k_brine > k_oil).all() and (k_oil > k_gas).all()
        assert (rho_brine > rho_oil).all() and (rho_oil > rho_gas).all()

    def test_gas_stiffens_with_depth(self):
        """The reason the table constants are not good enough: gas at 40 MPa is
        several times the modulus of gas at 10."""
        k_gas = fluid_suite(self.DEPTH)["gas"][0]
        assert (np.diff(k_gas) > 0).all()
        assert k_gas[-1] > 2.0 * k_gas[0]

    def test_the_parameters_can_be_overridden_and_are_reported_back(self):
        suite = fluid_suite(self.DEPTH, parameters={"salinity": 0.20})
        assert suite["parameters"]["salinity"] == pytest.approx(0.20)
        assert suite["parameters"]["api"] == FLUID_DEFAULTS["api"]   # the rest stand
        saltier = suite["brine"][1]
        assert (saltier > fluid_suite(self.DEPTH)["brine"][1]).all()

    def test_a_datum_shifts_the_whole_profile(self):
        """An offshore well's MD starts at a rig floor, not at the sea bed, so
        the gradients cannot start at zero depth."""
        shifted = fluid_suite(self.DEPTH, datum=100.0)
        assert shifted["pressure"] == pytest.approx([9.0, 19.0, 29.0, 39.0])

    def test_conditions_outside_the_fitted_range_are_reported(self):
        shallow = fluid_suite(np.array([100.0, 200.0]))
        assert shallow["warnings"], "an extrapolation must not pass silently"
        assert "outside the range" in next(iter(shallow["warnings"].values()))


class TestMixingTwoFluids:
    K_BRINE, RHO_BRINE = 2.7, 1.02
    K_GAS, RHO_GAS = 0.06, 0.20

    def _mix(self, sw, law="wood"):
        return mix_two_fluids(self.K_BRINE, self.RHO_BRINE, self.K_GAS,
                              self.RHO_GAS, sw, law=law)

    def test_the_end_members_are_the_fluids_themselves(self):
        k, rho = self._mix(np.array([1.0, 0.0]))
        assert k[0] == pytest.approx(self.K_BRINE)
        assert k[1] == pytest.approx(self.K_GAS)
        assert rho == pytest.approx([self.RHO_BRINE, self.RHO_GAS])

    def test_a_little_gas_dominates_under_wood(self):
        """The classic result: 10% gas takes a brine-filled pore most of the
        way to a gas-filled one."""
        k = self._mix(np.array([0.9]))[0][0]
        assert k < 0.2 * self.K_BRINE

    def test_patchy_is_the_stiff_limit_and_wood_the_soft_one(self):
        sw = np.linspace(0.05, 0.95, 10)
        soft = self._mix(sw, "wood")[0]
        stiff = self._mix(sw, "patchy")[0]
        assert (soft < stiff).all()

    def test_brie_sits_between_them_except_where_it_is_known_not_to(self):
        """`core.mixing.brie` says it: above roughly 90% gas the e = 3 curve
        crosses marginally below Wood's. It is an empirical fit, not a bound,
        and the vectorised form here must behave the same way."""
        sw = np.linspace(0.2, 0.95, 10)
        between = self._mix(sw, "brie")[0]
        assert (between >= self._mix(sw, "wood")[0] - 1e-9).all()
        assert (between <= self._mix(sw, "patchy")[0] + 1e-9).all()

        very_gassy = np.array([0.05])
        assert self._mix(very_gassy, "brie")[0][0] < self._mix(very_gassy, "wood")[0][0]

    def test_density_is_the_volume_average_under_every_law(self):
        for law in ("wood", "patchy", "brie"):
            rho = mix_two_fluids(self.K_BRINE, self.RHO_BRINE, self.K_GAS,
                                 self.RHO_GAS, [0.3], law=law)[1]
            assert rho[0] == pytest.approx(0.3 * self.RHO_BRINE + 0.7 * self.RHO_GAS)

    def test_it_follows_a_depth_varying_fluid(self):
        suite = fluid_suite(np.array([2000.0, 3000.0, 4000.0]))
        k_brine, rho_brine = suite["brine"]
        k_gas, rho_gas = suite["gas"]
        k, rho = mix_two_fluids(k_brine, rho_brine, k_gas, rho_gas, 0.2)
        assert k.shape == (3,)
        assert (k < k_brine).all()

    def test_saturations_outside_zero_to_one_are_clipped(self):
        k, _ = self._mix(np.array([-0.5, 1.5]))
        assert k[0] == pytest.approx(self.K_GAS)
        assert k[1] == pytest.approx(self.K_BRINE)

    def test_an_unknown_law_is_rejected(self):
        with pytest.raises(ValueError, match="unknown mixing law"):
            self._mix([0.5], law="magic")
