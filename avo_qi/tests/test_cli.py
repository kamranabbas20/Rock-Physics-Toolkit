"""The browser-free CLI: full analysis to disk, no server, no renderer.

The Streamlit app sends the values behind every plot to a browser to be drawn.
These tests pin the alternative — that the same analysis runs to completion
without opening a single outbound connection.
"""

from __future__ import annotations

import os
import socket

import numpy as np
import pandas as pd
import pytest

from avo_qi.cli import build_parser, main, run

DEMO = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "sample_data", "demo_well.las",
)


def parse(*argv):
    return build_parser().parse_args(list(argv))


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("cli") / "results")
    run(parse(DEMO, "--out", out, "--quiet", "--threshold", "0.02"))
    return out


class TestOutputs:
    EXPECTED = [
        "qc_curve_summary.csv", "qc_flags.csv", "qc_depth.csv",
        "well_standardised.csv", "well_time.csv", "reflectors.csv",
        "gather.csv", "gather.npy", "angle_stacks.csv", "report.txt",
    ]

    @pytest.mark.parametrize("filename", EXPECTED)
    def test_every_table_is_written(self, results, filename):
        path = os.path.join(results, filename)
        assert os.path.exists(path) and os.path.getsize(path) > 0

    def test_the_reflector_table_carries_the_analysis(self, results):
        table = pd.read_csv(os.path.join(results, "reflectors.csv"))
        for column in ("sample", "A_shuey", "B_shuey", "avo_class", "critical_angle"):
            assert column in table.columns
        assert (table["avo_class"] == "III").any()      # the demo gas sand

    def test_the_gather_matches_the_declared_angles(self, results):
        gather = np.load(os.path.join(results, "gather.npy"))
        frame = pd.read_csv(os.path.join(results, "gather.csv"))
        assert gather.shape[1] == 21                    # 0-40 in steps of 2
        assert gather.shape[0] == len(frame)
        assert np.isfinite(gather).all()

    def test_angle_stacks_are_written(self, results):
        stacks = pd.read_csv(os.path.join(results, "angle_stacks.csv"))
        assert "full" in stacks.columns and "TWT" in stacks.columns

    def test_the_report_summarises_the_run(self, results):
        report = open(os.path.join(results, "report.txt")).read()
        for line in ("well", "reflectors", "reflectivity method", "wavelet"):
            assert line in report
        assert "lithology" in report                    # the demo well has VSH

    def test_lithology_pairs_reach_the_table(self, results):
        table = pd.read_csv(os.path.join(results, "reflectors.csv"))
        assert "litho_pair" in table.columns
        assert "shale over sand" in set(table["litho_pair"])


class TestNoNetwork:
    """The reason this CLI exists."""

    def test_the_analysis_runs_with_every_connection_blocked(self, tmp_path, monkeypatch):
        attempts = []

        def blocked_connect(self, address, *args, **kwargs):
            attempts.append(address)
            raise OSError("connection blocked")

        def blocked_create(address, *args, **kwargs):
            attempts.append(address)
            raise OSError("connection blocked")

        monkeypatch.setattr(socket.socket, "connect", blocked_connect)
        monkeypatch.setattr(socket.socket, "connect_ex", blocked_connect)
        monkeypatch.setattr(socket, "create_connection", blocked_create)

        out = str(tmp_path / "offline")
        run(parse(DEMO, "--out", out, "--quiet", "--no-figures"))

        assert attempts == []
        assert os.path.exists(os.path.join(out, "reflectors.csv"))

    def test_the_cli_imports_no_web_stack(self):
        """Importing the CLI must not drag in Streamlit."""
        import subprocess
        import sys

        code = (
            "import sys; import avo_qi.cli; "
            "print('streamlit' in sys.modules or 'plotly' in sys.modules)"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                                text=True, cwd=os.path.dirname(os.path.dirname(
                                    os.path.dirname(os.path.abspath(__file__)))))
        assert result.stdout.strip() == "False", result.stderr


class TestOptions:
    def test_a_depth_window_narrows_the_output(self, tmp_path):
        wide = str(tmp_path / "wide")
        narrow = str(tmp_path / "narrow")
        run(parse(DEMO, "--out", wide, "--quiet"))
        run(parse(DEMO, "--out", narrow, "--quiet", "--top", "2030", "--base", "2100"))
        assert len(pd.read_csv(os.path.join(narrow, "well_standardised.csv"))) < \
            len(pd.read_csv(os.path.join(wide, "well_standardised.csv")))

    def test_a_fluid_case_can_be_chosen(self, tmp_path):
        brine = str(tmp_path / "brine")
        gas = str(tmp_path / "gas")
        run(parse(DEMO, "--out", brine, "--quiet", "--case", "brine", "--threshold", "0.02"))
        run(parse(DEMO, "--out", gas, "--quiet", "--case", "gas", "--threshold", "0.02"))
        a = pd.read_csv(os.path.join(brine, "reflectors.csv"))
        b = pd.read_csv(os.path.join(gas, "reflectors.csv"))

        def at_gas_sand_top(table):
            """The reflector nearest 2039.6 m — not the global minimum, which
            is the cemented streak's base in both cases."""
            return table.iloc[(table["depth"] - 2039.6).abs().argmin()]

        brine_top, gas_top = at_gas_sand_top(a), at_gas_sand_top(b)
        # Brine-filled the top is a hard event; gas-filled it is a Class III.
        assert brine_top["A_shuey"] > 0
        assert gas_top["A_shuey"] < 0
        assert gas_top["avo_class"] == "III"

    def test_an_unknown_case_is_rejected(self, tmp_path):
        with pytest.raises(SystemExit):
            run(parse(DEMO, "--out", str(tmp_path / "x"), "--quiet", "--case", "condensate"))

    def test_the_aki_richards_method_is_selectable(self, tmp_path):
        out = str(tmp_path / "ar")
        run(parse(DEMO, "--out", out, "--quiet", "--method", "aki_richards"))
        assert "aki_richards" in open(os.path.join(out, "report.txt")).read()

    def test_an_ormsby_wavelet_is_selectable(self, tmp_path):
        out = str(tmp_path / "orm")
        run(parse(DEMO, "--out", out, "--quiet", "--wavelet", "ormsby"))
        assert "ormsby" in open(os.path.join(out, "report.txt")).read()

    def test_despiking_is_reported(self, tmp_path):
        out = str(tmp_path / "spike")
        run(parse(DEMO, "--out", out, "--quiet", "--despike"))
        assert os.path.exists(os.path.join(out, "reflectors.csv"))

    def test_a_missing_curve_is_a_clean_error(self, tmp_path):
        thin = tmp_path / "thin.csv"
        pd.DataFrame({"DEPT": [1.0, 2.0], "GR": [50.0, 60.0]}).to_csv(thin, index=False)
        with pytest.raises(SystemExit) as err:
            run(parse(str(thin), "--out", str(tmp_path / "y"), "--quiet"))
        assert "missing required curve" in str(err.value)

    def test_main_returns_nonzero_on_a_bad_file(self, tmp_path):
        bad = tmp_path / "nope.las"
        bad.write_text("this is not a LAS file")
        assert main([str(bad), "--out", str(tmp_path / "z"), "--quiet"]) == 1


class TestFigures:
    def test_a_pdf_is_written_when_matplotlib_is_available(self, tmp_path):
        pytest.importorskip("matplotlib")
        out = str(tmp_path / "fig")
        run(parse(DEMO, "--out", out, "--quiet"))
        pdf = os.path.join(out, "figures.pdf")
        assert os.path.exists(pdf) and os.path.getsize(pdf) > 10_000

    def test_no_figures_skips_the_pdf(self, tmp_path):
        out = str(tmp_path / "nofig")
        run(parse(DEMO, "--out", out, "--quiet", "--no-figures"))
        assert not os.path.exists(os.path.join(out, "figures.pdf"))
        assert os.path.exists(os.path.join(out, "reflectors.csv"))
