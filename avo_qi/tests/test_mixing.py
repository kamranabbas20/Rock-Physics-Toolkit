"""Mixing laws for pore fluids and for mineral matrices."""

from __future__ import annotations

import numpy as np
import pytest

from avo_qi.core.mixing import (
    MINERAL_MIXING_LAWS,
    brie,
    fluid_density,
    fluid_mix,
    hashin_shtrikman_walpole,
    mineral_mix,
    patchy,
    wood,
)
from avo_qi.core.rockphysics import FLUIDS, MINERALS, hashin_shtrikman

K_BRINE, RHO_BRINE = FLUIDS["brine"]
K_OIL, RHO_OIL = FLUIDS["oil"]
K_GAS, RHO_GAS = FLUIDS["gas"]
QUARTZ = MINERALS["quartz"]
CLAY = MINERALS["clay"]
CALCITE = MINERALS["calcite"]


class TestFluidLimits:
    def test_a_single_phase_returns_itself(self):
        for law in ("wood", "patchy", "hill", "brie"):
            assert fluid_mix([K_BRINE], [1.0], law) == pytest.approx(K_BRINE)

    def test_wood_never_exceeds_patchy(self):
        for sg in np.linspace(0.0, 1.0, 21):
            sat = [1 - sg, sg]
            assert wood([K_BRINE, K_GAS], sat) <= patchy([K_BRINE, K_GAS], sat) + 1e-12

    def test_a_little_gas_collapses_wood_but_not_patchy(self):
        """The reason the choice of law matters at all."""
        sat = [0.95, 0.05]
        soft = wood([K_BRINE, K_GAS], sat)
        stiff = patchy([K_BRINE, K_GAS], sat)
        assert soft < K_BRINE / 5          # an order of magnitude down
        assert stiff > 0.9 * K_BRINE       # barely moved

    def test_the_end_members_agree_whatever_the_law(self):
        for law in ("wood", "patchy", "hill", "brie"):
            assert fluid_mix([K_BRINE, K_GAS], [1.0, 0.0], law) == pytest.approx(K_BRINE)
            assert fluid_mix([K_BRINE, K_GAS], [0.0, 1.0], law) == pytest.approx(K_GAS)

    def test_saturations_are_normalised(self):
        assert wood([K_BRINE, K_GAS], [80, 20]) == pytest.approx(
            wood([K_BRINE, K_GAS], [0.8, 0.2]))

    def test_negative_or_empty_saturations_are_rejected(self):
        with pytest.raises(ValueError):
            wood([K_BRINE, K_GAS], [-0.1, 1.1])
        with pytest.raises(ValueError):
            wood([K_BRINE, K_GAS], [0.0, 0.0])

    def test_mismatched_lengths_are_rejected(self):
        with pytest.raises(ValueError):
            wood([K_BRINE, K_GAS], [0.5, 0.3, 0.2])


class TestBrie:
    def test_exponent_one_is_the_patchy_limit(self):
        for s in (0.0, 0.3, 0.7, 1.0):
            assert brie(K_BRINE, K_GAS, s, exponent=1.0) == pytest.approx(
                patchy([K_BRINE, K_GAS], [s, 1 - s]))

    def test_larger_exponents_bend_towards_wood(self):
        s = 0.7
        soft = wood([K_BRINE, K_GAS], [s, 1 - s])
        values = [brie(K_BRINE, K_GAS, s, exponent=e) for e in (1, 2, 3, 5, 8)]
        assert all(b <= a + 1e-12 for a, b in zip(values, values[1:]))
        assert values[-1] > soft - 1e-9        # approaches but does not undercut

    def test_it_never_exceeds_the_patchy_limit(self):
        for s in np.linspace(0.0, 1.0, 21):
            assert brie(K_BRINE, K_GAS, s) <= patchy([K_BRINE, K_GAS], [s, 1 - s]) + 1e-9

    def test_it_sits_above_wood_across_the_useful_range(self):
        for s in np.linspace(0.15, 1.0, 18):
            assert brie(K_BRINE, K_GAS, s) >= wood([K_BRINE, K_GAS], [s, 1 - s]) - 1e-9

    def test_it_dips_just_below_wood_at_extreme_gas_saturation(self):
        """Brie is an empirical fit, not a rigorous bound.

        Above roughly 91% gas its curve crosses marginally under Wood's. The
        difference is tiny and both are near the gas modulus there, but the
        behaviour is real and worth pinning so it does not read as a bug.
        """
        s = 0.05                                   # 95% gas
        assert brie(K_BRINE, K_GAS, s) < wood([K_BRINE, K_GAS], [s, 1 - s])
        assert abs(brie(K_BRINE, K_GAS, s) - wood([K_BRINE, K_GAS], [s, 1 - s])) < 0.002

    def test_end_members(self):
        assert brie(K_BRINE, K_GAS, 1.0) == pytest.approx(K_BRINE)
        assert brie(K_BRINE, K_GAS, 0.0) == pytest.approx(K_GAS)

    def test_a_bad_exponent_is_rejected(self):
        with pytest.raises(ValueError):
            brie(K_BRINE, K_GAS, 0.5, exponent=0.0)

    def test_three_phases_mix_the_liquids_first(self):
        """Brie needs one gas and one liquid, so brine and oil mix by Wood."""
        value = fluid_mix([K_BRINE, K_OIL, K_GAS], [0.4, 0.4, 0.2], "brie")
        liquid = wood([K_BRINE, K_OIL], [0.5, 0.5])
        assert value == pytest.approx(brie(liquid, K_GAS, 0.8))


class TestFluidDensity:
    def test_it_is_the_volume_weighted_average(self):
        assert fluid_density([RHO_BRINE, RHO_GAS], [0.8, 0.2]) == pytest.approx(
            0.8 * RHO_BRINE + 0.2 * RHO_GAS)

    def test_it_does_not_depend_on_any_mixing_law(self):
        """Mass adds however the phases are arranged."""
        assert fluid_density([RHO_BRINE, RHO_OIL, RHO_GAS], [0.4, 0.4, 0.2]) == \
            pytest.approx(0.4 * RHO_BRINE + 0.4 * RHO_OIL + 0.2 * RHO_GAS)


class TestMineralBounds:
    K = [QUARTZ[0], CLAY[0], CALCITE[0]]
    G = [QUARTZ[1], CLAY[1], CALCITE[1]]
    RHO = [QUARTZ[2], CLAY[2], CALCITE[2]]
    F = [0.6, 0.3, 0.1]

    def test_the_bounds_nest_correctly(self):
        order = ["reuss", "hs_lower", "hs_average", "hs_upper", "voigt"]
        k = [mineral_mix(self.K, self.G, self.RHO, self.F, m)[0] for m in order]
        g = [mineral_mix(self.K, self.G, self.RHO, self.F, m)[1] for m in order]
        assert all(a <= b + 1e-9 for a, b in zip(k, k[1:])), k
        assert all(a <= b + 1e-9 for a, b in zip(g, g[1:])), g

    def test_hill_lies_between_voigt_and_reuss(self):
        voigt = mineral_mix(self.K, self.G, self.RHO, self.F, "voigt")[0]
        reuss = mineral_mix(self.K, self.G, self.RHO, self.F, "reuss")[0]
        hill = mineral_mix(self.K, self.G, self.RHO, self.F, "hill")[0]
        assert reuss < hill < voigt
        assert hill == pytest.approx(0.5 * (voigt + reuss))

    def test_density_is_the_same_under_every_law(self):
        expected = sum(f * r for f, r in zip(self.F, self.RHO))
        for method in MINERAL_MIXING_LAWS:
            assert mineral_mix(self.K, self.G, self.RHO, self.F, method)[2] == \
                pytest.approx(expected)

    def test_a_single_mineral_returns_itself(self):
        for method in MINERAL_MIXING_LAWS:
            k, g, rho = mineral_mix([QUARTZ[0]], [QUARTZ[1]], [QUARTZ[2]], [1.0], method)
            assert (k, g, rho) == pytest.approx(QUARTZ)

    def test_identical_minerals_collapse_the_bounds(self):
        bounds = hashin_shtrikman_walpole([36.6, 36.6], [45.0, 45.0], [0.5, 0.5])
        assert bounds["K_upper"] == pytest.approx(bounds["K_lower"])
        assert bounds["G_upper"] == pytest.approx(bounds["G_lower"])

    def test_the_n_phase_form_reduces_to_the_two_phase_one(self):
        """Walpole's extension must agree with the existing 2-phase bounds."""
        for f1 in (0.2, 0.5, 0.8):
            two = hashin_shtrikman(QUARTZ[0], QUARTZ[1], CLAY[0], CLAY[1],
                                   np.array([f1]))
            n = hashin_shtrikman_walpole([QUARTZ[0], CLAY[0]], [QUARTZ[1], CLAY[1]],
                                         [f1, 1 - f1])
            for key in ("K_upper", "K_lower", "G_upper", "G_lower"):
                assert n[key] == pytest.approx(float(two[key][0]), abs=1e-9)

    def test_it_handles_a_fluid_phase_with_no_shear(self):
        bounds = hashin_shtrikman_walpole([QUARTZ[0], K_BRINE], [QUARTZ[1], 0.0],
                                          [0.7, 0.3])
        assert bounds["G_lower"] == 0.0
        assert bounds["G_upper"] > 0.0

    def test_unknown_laws_are_rejected(self):
        with pytest.raises(ValueError):
            mineral_mix(self.K, self.G, self.RHO, self.F, "alchemy")
        with pytest.raises(ValueError):
            fluid_mix([K_BRINE, K_GAS], [0.8, 0.2], "osmosis")

    def test_mismatched_inputs_are_rejected(self):
        with pytest.raises(ValueError):
            mineral_mix(self.K, self.G[:2], self.RHO, self.F)
        with pytest.raises(ValueError):
            hashin_shtrikman_walpole([1.0, 2.0], [1.0], [0.5, 0.5])
