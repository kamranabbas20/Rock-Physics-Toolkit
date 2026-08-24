"""Lithology from VSH, and the lithology pair across a reflector."""

from __future__ import annotations

import numpy as np
import pytest

from avo_qi.core.lithology import (
    DEFAULT_VSH_CUTOFFS,
    GR_METHODS,
    LITHOLOGIES,
    UNDEFINED,
    classify_lithology,
    interface_lithology,
    lobe_lithology,
    lithology_fractions,
    vsh_from_gr,
)


class TestClassifyLithology:
    @pytest.mark.parametrize(
        "vsh,expected",
        [
            (0.00, "sand"), (0.10, "sand"), (0.15, "sand"),
            (0.16, "silty sand"), (0.35, "silty sand"),
            (0.36, "silt"), (0.60, "silt"),
            (0.61, "shale"), (1.00, "shale"),
        ],
    )
    def test_default_cutoffs(self, vsh, expected):
        assert classify_lithology([vsh])[0] == expected

    def test_cutoffs_are_inclusive_upper_bounds(self):
        """A sample exactly on a cutoff belongs to the cleaner class."""
        for name, bound in DEFAULT_VSH_CUTOFFS.items():
            assert classify_lithology([bound])[0] == name

    def test_missing_vsh_is_undefined_not_clean_sand(self):
        labels = classify_lithology([np.nan, np.inf, 0.05])
        assert labels[0] == UNDEFINED
        assert labels[1] == UNDEFINED
        assert labels[2] == "sand"

    def test_all_missing_returns_all_undefined(self):
        assert list(classify_lithology([np.nan, np.nan])) == [UNDEFINED, UNDEFINED]

    def test_custom_cutoffs_move_the_boundaries(self):
        strict = {"sand": 0.05, "silty sand": 0.20, "silt": 0.45}
        assert classify_lithology([0.10])[0] == "sand"                  # default
        assert classify_lithology([0.10], cutoffs=strict)[0] == "silty sand"

    def test_cutoffs_must_increase(self):
        with pytest.raises(ValueError):
            classify_lithology([0.2], cutoffs={"sand": 0.4, "silty sand": 0.2,
                                               "silt": 0.6})

    def test_incomplete_cutoffs_are_rejected(self):
        with pytest.raises(ValueError):
            classify_lithology([0.2], cutoffs={"sand": 0.15})

    def test_shape_is_preserved(self):
        vsh = np.linspace(0.0, 1.0, 37)
        assert classify_lithology(vsh).shape == vsh.shape

    def test_every_finite_value_lands_in_a_known_class(self):
        labels = classify_lithology(np.linspace(0.0, 1.0, 500))
        assert set(labels).issubset(set(LITHOLOGIES))


class TestLithologyFractions:
    def test_fractions_sum_to_one(self):
        labels = classify_lithology(np.linspace(0.0, 1.0, 200))
        fractions = lithology_fractions(labels)
        assert sum(fractions.values()) == pytest.approx(1.0)
        assert set(fractions) == set(LITHOLOGIES)

    def test_undefined_samples_are_excluded_from_the_denominator(self):
        labels = np.array(["sand", "shale", UNDEFINED], dtype=object)
        fractions = lithology_fractions(labels)
        assert fractions["sand"] == pytest.approx(0.5)
        assert fractions["shale"] == pytest.approx(0.5)

    def test_nothing_known_gives_zeros_not_a_crash(self):
        fractions = lithology_fractions(np.array([UNDEFINED], dtype=object))
        assert all(v == 0.0 for v in fractions.values())


class TestInterfaceLithology:
    LABELS = np.array(["shale", "shale", "sand", "sand", "shale"], dtype=object)

    def test_pairs_read_across_the_interface(self):
        found = interface_lithology(self.LABELS, [1, 3])
        assert list(found["upper"]) == ["shale", "sand"]
        assert list(found["lower"]) == ["sand", "shale"]
        assert list(found["pair"]) == ["shale over sand", "sand over shale"]

    def test_a_top_and_its_base_are_distinguishable(self):
        """The point of pairing: a sand top is not a sand base."""
        found = interface_lithology(self.LABELS, [1, 3])
        assert found["pair"][0] != found["pair"][1]

    def test_the_last_interface_clips_rather_than_overruns(self):
        found = interface_lithology(self.LABELS, [len(self.LABELS) - 1])
        assert found["lower"][0] == self.LABELS[-1]

    def test_empty_labels_are_safe(self):
        found = interface_lithology(np.array([], dtype=object), [0, 1])
        assert found["pair"].size == 0


class TestVshFromGr:
    GR = np.array([20.0, 40.0, 60.0, 80.0, 100.0, 120.0])

    def test_linear_index_spans_zero_to_one(self):
        vsh = vsh_from_gr(self.GR, 20.0, 120.0, method="linear")
        assert vsh[0] == pytest.approx(0.0)
        assert vsh[-1] == pytest.approx(1.0)
        assert np.all(np.diff(vsh) > 0)

    @pytest.mark.parametrize("method", ["larionov_tertiary", "larionov_older",
                                        "steiber", "clavier"])
    def test_non_linear_transforms_read_less_shale(self, method):
        linear = vsh_from_gr(self.GR, 20.0, 120.0, method="linear")
        other = vsh_from_gr(self.GR, 20.0, 120.0, method=method)
        assert np.all(other <= linear + 1e-9)
        assert other[0] == pytest.approx(0.0, abs=1e-6)

    def test_every_transform_stays_in_range(self):
        for method in GR_METHODS:
            vsh = vsh_from_gr(np.linspace(0.0, 200.0, 100), 20.0, 120.0, method=method)
            assert np.all(vsh >= 0.0) and np.all(vsh <= 1.0)

    def test_values_outside_the_references_are_clipped(self):
        vsh = vsh_from_gr([0.0, 500.0], 20.0, 120.0)
        assert vsh[0] == pytest.approx(0.0)
        assert vsh[1] == pytest.approx(1.0)

    def test_references_default_to_percentiles(self):
        gr = np.concatenate([np.full(50, 25.0), np.full(50, 115.0)])
        vsh = vsh_from_gr(gr)
        assert vsh[0] == pytest.approx(0.0, abs=0.05)
        assert vsh[-1] == pytest.approx(1.0, abs=0.05)

    def test_identical_references_are_rejected(self):
        with pytest.raises(ValueError):
            vsh_from_gr(self.GR, 50.0, 50.0)

    def test_unknown_method_is_rejected(self):
        with pytest.raises(ValueError):
            vsh_from_gr(self.GR, 20.0, 120.0, method="guesswork")

    def test_an_all_nan_log_returns_nan(self):
        assert np.all(np.isnan(vsh_from_gr([np.nan, np.nan])))


class TestAgainstTheDemoWell:
    def test_demo_layers_classify_as_intended(self):
        import os

        from avo_qi.io.loader import read_well, standardise

        las = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "sample_data", "demo_well.las",
        )
        raw, units = read_well(las)
        df = standardise(raw, units=units).df
        labels = classify_lithology(df["VSH"].to_numpy(float))
        depth = df["DEPTH"].to_numpy(float)

        shale = (depth >= 2005) & (depth < 2035)
        sand = (depth >= 2045) & (depth < 2065)
        assert set(labels[shale]) == {"shale"}       # VSH 0.85
        assert set(labels[sand]) == {"sand"}         # VSH 0.10

    def test_gas_sand_top_is_a_shale_over_sand_pair(self):
        labels = np.array(["shale"] * 40 + ["sand"] * 30, dtype=object)
        found = interface_lithology(labels, [39])
        assert found["pair"][0] == "shale over sand"


class TestLobeLithology:
    """Name the rock the AVO was actually fitted to.

    Once the elastic properties come from averaging a half-lobe apiece, the two
    samples nearest the extremum are no longer the layers being described.
    """

    @staticmethod
    def _bounds(upper, lower, resolved=True):
        return {"upper_start": np.array([upper[0]]),
                "upper_stop": np.array([upper[1]]),
                "lower_start": np.array([lower[0]]),
                "lower_stop": np.array([lower[1]]),
                "resolved": np.array([resolved])}

    def test_it_reads_the_commonest_label_over_each_half(self):
        labels = np.array(["shale"] * 20 + ["sand"] * 20, dtype=object)
        found = lobe_lithology(labels, self._bounds((12, 20), (19, 28)))
        assert found["pair"][0] == "shale over sand"

    def test_a_stray_sample_does_not_rename_the_layer(self):
        """The failure this pins: one sample of silt inside eight of shale
        named the pair from the silt, because the boundary reading looks at
        exactly the two samples the wave averaged away."""
        labels = np.array(["shale"] * 20 + ["sand"] * 20, dtype=object)
        labels[19] = "silt"                       # the sample at the boundary
        labels[20] = "silt"
        assert interface_lithology(labels, [19])["pair"][0] == "silt over silt"
        found = lobe_lithology(labels, self._bounds((12, 20), (19, 28)))
        assert found["pair"][0] == "shale over sand"

    def test_undefined_samples_are_ignored_unless_the_half_is_all_undefined(self):
        labels = np.array([UNDEFINED] * 20 + ["sand"] * 20, dtype=object)
        labels[15:20] = "shale"
        found = lobe_lithology(labels, self._bounds((10, 20), (19, 28)))
        assert found["pair"][0] == "shale over sand"

        blank = np.full(40, UNDEFINED, dtype=object)
        found = lobe_lithology(blank, self._bounds((10, 20), (19, 28)))
        assert found["pair"][0] == f"{UNDEFINED} over {UNDEFINED}"

    def test_an_unresolved_lobe_falls_back_to_the_interface(self):
        labels = np.array(["shale"] * 20 + ["sand"] * 20, dtype=object)
        found = lobe_lithology(labels, self._bounds((0, 0), (0, 0), resolved=False),
                               samples=[19])
        assert found["pair"][0] == "shale over sand"

    def test_no_labels_at_all_is_not_an_error(self):
        found = lobe_lithology(np.array([], dtype=object),
                               self._bounds((0, 1), (1, 2)))
        assert found["pair"].size == 0
