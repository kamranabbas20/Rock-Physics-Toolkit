"""Porosity and saturation from raw logs.

These transforms are a *fallback*. A well that arrives with a petrophysical
interpretation should keep it — it was made with core, pressures and local
calibration none of this has. What is tested here is that the fallback is
honest: it says when its assumptions are failing rather than clipping the
evidence away, and the shaly-sand model actually reduces to the clean-sand one.
"""

from __future__ import annotations

import numpy as np
import pytest

from avo_qi.core.petrophysics import (
    FLUID_DENSITY,
    MATRIX_DENSITY,
    effective_porosity,
    porosity_from_density,
    porosity_from_density_neutron,
    sw_archie,
    sw_simandoux,
)


class TestDensityPorosity:
    def test_the_end_members_are_exact(self):
        found = porosity_from_density([2.65, 1.0, 1.825], 2.65, 1.0)["phi"]
        assert found == pytest.approx([0.0, 1.0, 0.5])

    def test_it_is_a_straight_line_in_density(self):
        rhob = np.linspace(1.9, 2.65, 20)
        phi = porosity_from_density(rhob)["phi"]
        assert (np.diff(phi) < 0).all()
        assert np.allclose(np.diff(phi, 2), 0.0, atol=1e-12)

    def test_the_wrong_matrix_density_shifts_everything_one_way(self):
        """The failure mode worth knowing about: it is not noise, it is a bias
        in the same direction at every sample."""
        rhob = np.linspace(2.0, 2.5, 50)
        quartz = porosity_from_density(rhob, MATRIX_DENSITY["sandstone (quartz)"])["phi"]
        dolomite = porosity_from_density(rhob, MATRIX_DENSITY["dolomite"])["phi"]
        assert (dolomite > quartz).all()
        assert np.median(dolomite - quartz) > 0.08

    def test_it_reports_how_much_it_had_to_clip(self):
        """A heavily clipped result is the matrix density saying it is wrong,
        which the caller has to be able to see."""
        rhob = np.array([2.80, 2.75, 2.40, 2.30])       # two samples denser than quartz
        found = porosity_from_density(rhob, 2.65, 1.0)
        assert found["clipped"] == pytest.approx(0.5)
        assert (found["phi"] >= 0).all()
        raw = porosity_from_density(rhob, 2.65, 1.0, clip=False)["phi"]
        assert (raw[:2] < 0).all()

    def test_gas_in_the_pore_reads_higher_porosity(self):
        rhob = np.full(3, 2.2)
        brine = porosity_from_density(rhob, 2.65, FLUID_DENSITY["fresh mud filtrate"])["phi"]
        gas = porosity_from_density(rhob, 2.65, FLUID_DENSITY["gas"])["phi"]
        assert (gas < brine).all()      # a lighter assumed fluid needs less pore

    def test_nulls_stay_null(self):
        phi = porosity_from_density([2.3, np.nan], 2.65, 1.0)["phi"]
        assert np.isnan(phi[1]) and np.isfinite(phi[0])

    def test_an_impossible_pair_is_rejected(self):
        with pytest.raises(ValueError, match="must differ"):
            porosity_from_density([2.3], 2.65, 2.65)


class TestDensityNeutronPorosity:
    def test_they_agree_in_a_liquid_filled_hole(self):
        """No gas: both curves read the same porosity and any combination of
        them returns it."""
        phi_true = 0.25
        rhob = 2.65 - phi_true * (2.65 - 1.0)
        for method in ("rms", "average"):
            found = porosity_from_density_neutron([rhob], [phi_true], method=method)
            assert found["phi"][0] == pytest.approx(phi_true)
            assert found["separation"][0] == pytest.approx(0.0, abs=1e-12)

    def test_gas_separates_the_two_curves(self):
        """The crossover: density porosity reads high, neutron reads low."""
        found = porosity_from_density_neutron([2.15], [0.10])
        assert found["phi_density"][0] > found["phi_neutron"][0]
        assert found["separation"][0] > 0.15
        # ...and the RMS sits above the average, nearer the density reading.
        average = porosity_from_density_neutron([2.15], [0.10], method="average")
        assert found["phi"][0] > average["phi"][0]

    def test_a_neutron_log_in_percent_is_recognised(self):
        as_fraction = porosity_from_density_neutron([2.3], [0.20])
        as_percent = porosity_from_density_neutron([2.3], [20.0])
        assert as_percent["phi"][0] == pytest.approx(as_fraction["phi"][0])

    def test_an_unknown_method_is_rejected(self):
        with pytest.raises(ValueError, match="unknown method"):
            porosity_from_density_neutron([2.3], [0.2], method="magic")


class TestEffectivePorosity:
    def test_the_shale_bound_part_is_removed(self):
        assert effective_porosity([0.30], [0.5], 0.10)[0] == pytest.approx(0.25)

    def test_clean_rock_is_untouched(self):
        assert effective_porosity([0.30], [0.0], 0.10)[0] == pytest.approx(0.30)

    def test_it_cannot_go_negative(self):
        assert effective_porosity([0.05], [1.0], 0.30)[0] == pytest.approx(0.0)


class TestArchie:
    def test_the_textbook_case(self):
        """φ = 0.25, Rw = 0.05, Rt = 3.2, a = 1, m = n = 2 gives Sw = 0.5."""
        sw = sw_archie([3.2], [0.25], rw=0.05, a=1.0, m=2.0, n=2.0)
        assert sw[0] == pytest.approx(0.5)

    def test_more_resistive_rock_holds_less_water(self):
        sw = sw_archie([1.0, 10.0, 100.0], [0.25, 0.25, 0.25], rw=0.05)
        assert (np.diff(sw) < 0).all()

    def test_tighter_rock_at_the_same_resistivity_holds_more_water(self):
        tight = sw_archie([20.0], [0.10], rw=0.05)[0]
        porous = sw_archie([20.0], [0.30], rw=0.05)[0]
        assert tight > porous

    def test_saturation_stays_within_its_bounds(self):
        sw = sw_archie([0.01, 1e6], [0.25, 0.25], rw=0.05)
        assert sw[0] == 1.0 and 0.0 <= sw[1] <= 1.0

    def test_a_zero_porosity_sample_is_null_not_infinite(self):
        sw = sw_archie([10.0], [0.0], rw=0.05, clip=False)
        assert not np.isfinite(sw[0])

    def test_a_per_sample_rw_works(self):
        """Rw varies with temperature, so it is a curve as often as a number."""
        sw = sw_archie([3.2, 3.2], [0.25, 0.25], rw=[0.05, 0.02])
        assert sw[0] > sw[1]


class TestSimandoux:
    def test_it_reduces_to_archie_in_clean_rock(self):
        """A shaly-sand model that does not is not a correction to anything."""
        rt = np.array([1.0, 3.2, 20.0, 100.0])
        phi = np.full(4, 0.25)
        archie = sw_archie(rt, phi, rw=0.05, a=1.0, m=2.0, n=2.0)
        shaly = sw_simandoux(rt, phi, np.zeros(4), rw=0.05, r_shale=2.0,
                             a=1.0, m=2.0)
        assert shaly == pytest.approx(archie, abs=1e-9)

    def test_shale_conductivity_lowers_the_water_it_reads(self):
        """Archie blames every conductive path on water and reads too much of
        it; taking the clay's own conduction out is what recovers the pay."""
        rt, phi = np.full(3, 4.0), np.full(3, 0.22)
        vsh = np.array([0.0, 0.25, 0.50])
        sw = sw_simandoux(rt, phi, vsh, rw=0.05, r_shale=1.5)
        assert (np.diff(sw) < 0).all()
        assert sw[0] > sw[2] + 0.1

    def test_a_more_resistive_shale_matters_less(self):
        common = dict(rw=0.05, a=1.0, m=2.0)
        conductive = sw_simandoux([4.0], [0.22], [0.4], r_shale=0.5, **common)[0]
        resistive = sw_simandoux([4.0], [0.22], [0.4], r_shale=50.0, **common)[0]
        archie = sw_archie([4.0], [0.22], n=2.0, **common)[0]
        assert conductive < resistive
        assert resistive == pytest.approx(archie, abs=0.02)

    def test_it_stays_within_bounds_and_rejects_a_nonsense_shale(self):
        assert 0.0 <= sw_simandoux([0.01], [0.25], [0.3])[0] <= 1.0
        with pytest.raises(ValueError, match="positive"):
            sw_simandoux([4.0], [0.25], [0.3], r_shale=0.0)


class TestAgainstARealWell:
    """15/9-19-A carries both the raw logs and someone else's interpretation of
    them, which is the only honest way to check a first pass: not against a
    textbook number, but against what a petrophysicist made of the same rock."""

    @staticmethod
    def _well():
        import os

        from avo_qi.io.loader import read_well

        here = os.path.dirname(os.path.abspath(__file__))
        frame, _ = read_well(os.path.join(here, "data", "15_9_19_A.las"))
        return frame.replace(-999.25, np.nan)

    def test_density_porosity_tracks_the_files_own_porosity(self):
        frame = self._well()
        mine = porosity_from_density(frame["RHOB"].to_numpy(float),
                                     rho_matrix=2.65, rho_fluid=1.0)["phi"]
        theirs = frame["PHIF"].to_numpy(float)
        both = np.isfinite(mine) & np.isfinite(theirs)
        assert both.sum() > 500
        # Not identical — theirs is shale-corrected and calibrated — but the
        # same rock, so they must correlate strongly and agree in the mean.
        assert np.corrcoef(mine[both], theirs[both])[0, 1] > 0.8
        assert abs(np.median(mine[both] - theirs[both])) < 0.10

    def test_archie_tracks_the_files_own_saturation(self):
        frame = self._well()
        phi = frame["PHIF"].to_numpy(float)
        rt = frame["RT"].to_numpy(float)
        rw = np.nanmedian(frame["RW"].to_numpy(float))
        mine = sw_archie(rt, phi, rw=rw, m=np.nanmedian(frame["M"].to_numpy(float)),
                         n=np.nanmedian(frame["N"].to_numpy(float)))
        theirs = frame["SW"].to_numpy(float)
        both = np.isfinite(mine) & np.isfinite(theirs)
        assert both.sum() > 500
        assert np.corrcoef(mine[both], theirs[both])[0, 1] > 0.8

    def test_the_matrix_density_the_file_used_is_not_the_default(self):
        """Which is the point of asking rather than assuming: this well's own
        RHOMA curve is what its porosity was made with."""
        frame = self._well()
        rhoma = frame["RHOMA"].to_numpy(float)
        assert np.isfinite(rhoma).any()
        assert abs(float(np.nanmedian(rhoma)) - 2.65) > 0.0
