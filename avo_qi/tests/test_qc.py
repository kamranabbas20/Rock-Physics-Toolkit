"""Log QC: nulls, ranges, physical consistency, spikes and the depth axis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from avo_qi.core.qc import (
    DEFAULT_RANGES,
    LAS_NULLS,
    completeness,
    curve_summary,
    depth_qc,
    despike,
    elastic_flags,
    qc_flags,
    range_flags,
    replace_nulls,
    spike_flags,
)


def clean_logs(n=400, seed=0):
    rng = np.random.default_rng(seed)
    vp = np.full(n, 2400.0) + rng.normal(0, 8, n)
    vs = np.full(n, 1200.0) + rng.normal(0, 5, n)
    rho = np.full(n, 2.35) + rng.normal(0, 0.006, n)
    return vp, vs, rho


class TestReplaceNulls:
    @pytest.mark.parametrize("null", LAS_NULLS)
    def test_every_known_sentinel_becomes_nan(self, null):
        values = np.array([2400.0, null, 2450.0])
        assert np.isnan(replace_nulls(values)[1])

    def test_real_values_survive(self):
        values = np.array([2400.0, -999.25, 2450.0])
        out = replace_nulls(values)
        assert out[0] == 2400.0 and out[2] == 2450.0

    def test_a_header_null_can_be_supplied(self):
        values = np.array([2400.0, -1234.5])
        assert np.isnan(replace_nulls(values, extra=-1234.5)[1])

    def test_the_input_is_not_modified_in_place(self):
        values = np.array([2400.0, -999.25])
        replace_nulls(values)
        assert values[1] == -999.25

    def test_a_non_finite_header_null_is_ignored(self):
        out = replace_nulls(np.array([2400.0, 2450.0]), extra=np.nan)
        assert np.all(np.isfinite(out))


class TestRangeFlags:
    def test_flags_only_outside_the_range(self):
        values = np.array([1000.0, 2400.0, 8000.0])
        flags = range_flags(values, *DEFAULT_RANGES["VP"])
        assert list(flags) == [True, False, True]

    def test_missing_data_is_not_flagged_as_wrong(self):
        """A NaN is absent, not out of range."""
        assert not range_flags(np.array([np.nan]), 1400.0, 7000.0)[0]

    def test_bounds_are_inclusive(self):
        lo, hi = DEFAULT_RANGES["VP"]
        assert not range_flags(np.array([lo, hi]), lo, hi).any()


class TestElasticFlags:
    def test_shear_faster_than_compressional_is_caught(self):
        vp, vs, rho = clean_logs()
        vs[200] = 2600.0
        assert elastic_flags(vp, vs, rho)["vs_exceeds_vp"][200]

    def test_negative_poisson_is_caught(self):
        vp, vs, rho = clean_logs()
        vs[210] = 1900.0                       # Vp/Vs = 1.26, below sqrt(2)
        flags = elastic_flags(vp, vs, rho)
        assert flags["vpvs_below_limit"][210]

    def test_clean_logs_raise_nothing(self):
        vp, vs, rho = clean_logs()
        flags = elastic_flags(vp, vs, rho)
        assert not any(mask.any() for mask in flags.values())

    def test_non_positive_values_are_caught(self):
        vp, vs, rho = clean_logs()
        vp[5] = 0.0
        rho[6] = -1.0
        flags = elastic_flags(vp, vs, rho)
        assert flags["non_positive"][5] and flags["non_positive"][6]

    def test_missing_samples_are_not_flagged(self):
        vp, vs, rho = clean_logs()
        vp[10] = np.nan
        flags = elastic_flags(vp, vs, rho)
        assert not any(mask[10] for mask in flags.values())

    def test_poisson_approaches_but_never_reaches_a_half(self):
        """A very slow shear log is suspicious but not *impossible*, so it is
        left to the range and Vp/Vs checks rather than flagged as unphysical."""
        flags = elastic_flags(np.array([2400.0]), np.array([1.0]), np.array([2.35]))
        assert not flags["poisson_out_of_range"][0]

    def test_zero_shear_hits_the_fluid_limit_exactly(self):
        """Vs = 0 gives Poisson exactly 0.5 — a fluid, not a rock."""
        flags = elastic_flags(np.array([1500.0]), np.array([0.0]), np.array([1.02]))
        assert flags["poisson_out_of_range"][0]


class TestDepthQc:
    def test_a_clean_axis_passes(self):
        depth = np.arange(400) * 0.1524 + 2000.0
        report = depth_qc(depth)
        assert report["monotonic_increasing"]
        assert report["duplicates"] == 0
        assert not report["step_varies"]
        assert report["step"] == pytest.approx(0.1524)
        assert report["gaps"] == 0

    def test_duplicates_break_monotonicity(self):
        depth = np.arange(400) * 0.1524 + 2000.0
        depth[300] = depth[299]
        report = depth_qc(depth)
        assert report["duplicates"] == 1
        assert not report["monotonic_increasing"]

    def test_a_gap_is_counted(self):
        depth = np.concatenate([np.arange(100) * 0.1524,
                                np.arange(100) * 0.1524 + 50.0])
        assert depth_qc(depth)["gaps"] == 1

    def test_start_and_stop_are_reported(self):
        depth = np.arange(10) * 0.5 + 2000.0
        report = depth_qc(depth)
        assert report["start"] == pytest.approx(2000.0)
        assert report["stop"] == pytest.approx(2004.5)

    def test_an_empty_axis_is_safe(self):
        report = depth_qc(np.array([]))
        assert report["n_samples"] == 0
        assert np.isnan(report["start"])


class TestSpikes:
    def test_an_isolated_spike_is_found(self):
        vp, _, _ = clean_logs()
        vp[120] = 6800.0
        assert spike_flags(vp)[120]

    def test_a_clean_log_has_no_spikes(self):
        vp, _, _ = clean_logs()
        assert not spike_flags(vp).any()

    def test_a_real_step_is_not_a_spike(self):
        """A layer boundary is a step, not a spike, and must survive."""
        vp = np.concatenate([np.full(200, 2400.0), np.full(200, 3000.0)])
        flagged = spike_flags(vp, window=11)
        assert flagged.sum() <= 2          # at most the samples on the step itself

    def test_despike_repairs_only_what_it_flagged(self):
        vp, _, _ = clean_logs()
        original = vp.copy()
        vp[120] = 6800.0
        repaired, flags = despike(vp)
        assert repaired[120] == pytest.approx(original[120], abs=60)
        untouched = ~flags
        assert np.allclose(repaired[untouched], vp[untouched], equal_nan=True)

    def test_despiking_a_clean_log_changes_nothing(self):
        vp, _, _ = clean_logs()
        repaired, flags = despike(vp)
        assert not flags.any()
        assert np.allclose(repaired, vp)

    def test_missing_samples_survive_despiking(self):
        vp, _, _ = clean_logs()
        vp[50] = np.nan
        repaired, _ = despike(vp)
        assert np.isnan(repaired[50])

    def test_short_and_constant_logs_are_safe(self):
        assert not spike_flags(np.array([1.0, 2.0])).any()
        assert not spike_flags(np.full(50, 2400.0)).any()

    def test_a_higher_threshold_flags_no_more(self):
        vp, _, _ = clean_logs()
        vp[120] = 6800.0
        assert spike_flags(vp, threshold=10.0).sum() <= spike_flags(vp, threshold=3.0).sum()


class TestSummaryAndFlags:
    def dirty_frame(self):
        vp, vs, rho = clean_logs()
        vp[50] = np.nan
        vp[121] = 900.0                       # out of range
        vs[200] = 2600.0                      # faster than Vp
        return pd.DataFrame({"VP": vp, "VS": vs, "RHOB": rho})

    def test_summary_reports_coverage_and_range_breaches(self):
        summary = curve_summary(self.dirty_frame()).set_index("curve")
        assert summary.loc["VP", "n_missing"] == 1
        assert summary.loc["VP", "out_of_range"] >= 1
        assert summary.loc["RHOB", "out_of_range"] == 0
        assert summary.loc["VP", "present"] < 1.0

    def test_summary_carries_the_expected_range(self):
        summary = curve_summary(self.dirty_frame()).set_index("curve")
        assert summary.loc["VP", "expected_min"] == DEFAULT_RANGES["VP"][0]

    def test_unknown_curves_get_no_range_check(self):
        frame = pd.DataFrame({"MYSTERY": [1.0, 2.0, 3.0]})
        summary = curve_summary(frame).set_index("curve")
        assert summary.loc["MYSTERY", "out_of_range"] == 0
        assert np.isnan(summary.loc["MYSTERY", "expected_min"])

    def test_flags_frame_covers_every_check(self):
        flags = qc_flags(self.dirty_frame())
        for column in ("VP_out_of_range", "VP_spike", "vs_exceeds_vp", "any_flag"):
            assert column in flags.columns
        assert flags["any_flag"].sum() >= 2

    def test_a_clean_well_raises_no_flags(self):
        vp, vs, rho = clean_logs()
        flags = qc_flags(pd.DataFrame({"VP": vp, "VS": vs, "RHOB": rho}))
        assert not flags["any_flag"].any()

    def test_flags_align_with_the_frame(self):
        frame = self.dirty_frame()
        assert len(qc_flags(frame)) == len(frame)

    def test_partial_wells_are_handled(self):
        """A well with no Vs still gets range and spike checks on what it has."""
        vp, _, rho = clean_logs()
        flags = qc_flags(pd.DataFrame({"VP": vp, "RHOB": rho}))
        assert "VP_out_of_range" in flags.columns
        assert "vs_exceeds_vp" not in flags.columns

    def test_completeness(self):
        assert completeness(np.array([1.0, np.nan, 3.0])) == pytest.approx(2 / 3)
        assert completeness(np.array([])) == 0.0


class TestAgainstTheDemoWell:
    def test_the_demo_well_passes_qc_cleanly(self):
        import os

        from avo_qi.io.loader import read_well, standardise

        las = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "sample_data", "demo_well.las",
        )
        raw, units = read_well(las)
        well = standardise(raw, units=units)
        flags = qc_flags(well.df[["VP", "VS", "RHOB"]])
        # Layer boundaries are steps, so allow a couple of samples on each.
        assert flags["any_flag"].sum() <= 12
        assert not flags["vs_exceeds_vp"].any()
        assert not flags["VP_out_of_range"].any()


@pytest.fixture(scope="module")
def realistic_well(tmp_path_factory):
    """A forged LAS with the defects a real one arrives with."""
    from avo_qi.io.loader import read_well, standardise

    path = TestAgainstARealisticWell.build(tmp_path_factory.mktemp("realish"))
    raw, units = read_well(path)
    return standardise(raw, units=units, name="REALISH-1")


class TestAgainstARealisticWell:
    """A forged LAS with the defects a real one arrives with: sonic in us/ft,
    density in kg/m3, null intervals, spikes and a bad shear section."""

    @staticmethod
    def build(tmp_path):
        import lasio
        from scipy.ndimage import uniform_filter1d

        rng = np.random.default_rng(3)
        depth = np.arange(1800.0, 2400.0, 0.1524)
        n = depth.size
        vp = np.full(n, 2600.0); vs = np.full(n, 1300.0)
        rho = np.full(n, 2.40); gr = np.full(n, 90.0)
        for lo, hi, v, s, r, g in ((2050, 2085, 2250.0, 1400.0, 2.18, 28.0),
                                   (2120, 2150, 2750.0, 1520.0, 2.36, 35.0)):
            m = (depth >= lo) & (depth < hi)
            vp[m], vs[m], rho[m], gr[m] = v, s, r, g
        for a in (vp, vs, rho, gr):           # gradational boundaries
            a[:] = uniform_filter1d(a, size=9)
        vp += rng.normal(0, 12, n); vs += rng.normal(0, 9, n)
        rho += rng.normal(0, 0.01, n); gr += rng.normal(0, 4, n)

        vs[:120] = -999.25                    # no shear over the top
        vp[900:940] = -999.25                 # washout
        rho[900:940] = -999.25
        vp[1500] = 7900.0                     # spikes
        rho[1600] = 0.9
        vs[2000:2040] = vp[2000:2040] * 0.95  # bad shear: Vs approaches Vp

        las = lasio.LASFile()
        las.well["NULL"] = lasio.HeaderItem("NULL", value=-999.25)
        las.append_curve("DEPT", depth, unit="M")
        las.append_curve("DTCO", np.where(vp > 0, 1e6 / (vp / 0.3048), -999.25),
                         unit="US/F")
        las.append_curve("DTSM", np.where(vs > 0, 1e6 / (vs / 0.3048), -999.25),
                         unit="US/F")
        las.append_curve("RHOZ", np.where(rho > 0, rho * 1000.0, -999.25), unit="KG/M3")
        las.append_curve("GR", gr, unit="GAPI")
        path = str(tmp_path / "realish.las")
        las.write(path, version=2.0, fmt="%.4f")
        return path

    def test_sonic_and_density_are_recognised_and_converted(self, realistic_well):
        assert realistic_well.mapping["VP"] == "DTCO"
        assert realistic_well.mapping["VS"] == "DTSM"
        assert realistic_well.mapping["RHOB"] == "RHOZ"
        vp = realistic_well.df["VP"].to_numpy(float)
        rho = realistic_well.df["RHOB"].to_numpy(float)
        # Converted to m/s and g/cc, not left as slowness and kg/m3.
        assert 2000 < np.nanmedian(vp) < 3500
        assert 2.0 < np.nanmedian(rho) < 2.6

    def test_load_notes_are_not_duplicated(self, realistic_well):
        assert len(realistic_well.notes) == len(set(realistic_well.notes))

    def test_null_intervals_become_missing(self, realistic_well):
        frame = realistic_well.df.copy()
        for column in ("VP", "VS", "RHOB"):
            frame[column] = replace_nulls(frame[column].to_numpy(float))
        summary = curve_summary(frame[["VP", "VS", "RHOB"]]).set_index("curve")
        assert summary.loc["VS", "n_missing"] == 120      # no shear over the top
        assert summary.loc["VP", "n_missing"] == 40       # washout
        assert summary.loc["RHOB", "n_missing"] == 40

    def test_spikes_and_range_breaches_are_caught(self, realistic_well):
        frame = realistic_well.df.copy()
        for column in ("VP", "VS", "RHOB"):
            frame[column] = replace_nulls(frame[column].to_numpy(float))
        flags = qc_flags(frame)
        assert flags["VP_out_of_range"].sum() == 1        # the 7900 m/s spike
        assert flags["RHOB_out_of_range"].sum() == 1      # the 0.9 g/cc spike
        assert flags["VP_spike"].sum() >= 1
        assert flags["RHOB_spike"].sum() >= 1

    def test_the_bad_shear_interval_is_caught(self, realistic_well):
        frame = realistic_well.df.copy()
        for column in ("VP", "VS", "RHOB"):
            frame[column] = replace_nulls(frame[column].to_numpy(float))
        flags = qc_flags(frame)
        # Vs at 0.95 Vp is below sqrt(2), so Poisson goes negative.
        assert flags["vpvs_below_limit"].sum() == 40
        assert flags["poisson_out_of_range"].sum() == 40

    def test_most_of_the_well_is_still_usable(self, realistic_well):
        frame = realistic_well.df.copy()
        for column in ("VP", "VS", "RHOB"):
            frame[column] = replace_nulls(frame[column].to_numpy(float))
        flags = qc_flags(frame)
        assert flags["any_flag"].mean() < 0.05            # ~1% here
