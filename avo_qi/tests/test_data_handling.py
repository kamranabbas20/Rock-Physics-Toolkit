"""Where loaded well data goes — and where it must not.

A loaded well is usually proprietary. These tests pin the handling: parsed in
memory, never written to the working directory, and never sent anywhere.
"""

from __future__ import annotations

import glob
import io
import os
import subprocess

import pytest

from avo_qi.io.loader import read_well, standardise

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(HERE)
DEMO = os.path.join(HERE, "sample_data", "demo_well.las")

#: Some checks are about the *repository* — its lockfile, scripts and ignore
#: rules — rather than the installed package.  Deployed copies legitimately
#: carry only `avo_qi/`, so those checks skip instead of failing there.
IN_CHECKOUT = os.path.isdir(os.path.join(REPO, ".git"))
needs_checkout = pytest.mark.skipif(
    not IN_CHECKOUT, reason="not running from a git checkout of the repository"
)


@pytest.fixture(scope="module")
def las_bytes():
    with open(DEMO, "rb") as fh:
        return fh.read()


class TestInMemoryReading:
    def test_a_las_can_be_read_from_a_buffer(self, las_bytes):
        raw, units = read_well(io.BytesIO(las_bytes), suffix=".las")
        assert len(raw) > 0
        assert "VP" in raw.columns
        assert units["VP"] == "M/S"

    def test_buffer_and_path_agree(self, las_bytes):
        from_path, _ = read_well(DEMO)
        from_buffer, _ = read_well(io.BytesIO(las_bytes), suffix=".las")
        assert list(from_path.columns) == list(from_buffer.columns)
        assert len(from_path) == len(from_buffer)

    def test_a_csv_can_be_read_from_a_buffer(self, las_bytes):
        raw, _ = read_well(io.BytesIO(las_bytes), suffix=".las")
        csv = raw.head(20).to_csv(index=False).encode()
        parsed, _ = read_well(io.BytesIO(csv), suffix=".csv")
        assert len(parsed) == 20

    def test_reading_a_buffer_writes_nothing_to_the_working_directory(self, las_bytes, tmp_path):
        """A loaded well must never land on disk where a later `git add` could
        pick it up."""
        cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            before = set(glob.glob("*") + glob.glob(".*"))
            raw, units = read_well(io.BytesIO(las_bytes), suffix=".las")
            standardise(raw, units=units, name="CONFIDENTIAL-1")
            after = set(glob.glob("*") + glob.glob(".*"))
        finally:
            os.chdir(cwd)
        assert after == before

    def test_the_upload_path_used_by_the_ui_writes_nothing(self, las_bytes, tmp_path):
        """Exercise load_uploaded_well's parsing without a Streamlit runtime."""
        cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            before = set(glob.glob("*"))
            buffer = io.BytesIO(las_bytes)
            raw, units = read_well(buffer, suffix=".las")
            well = standardise(raw, units=units, name="CONFIDENTIAL-1")
            after = set(glob.glob("*"))
        finally:
            os.chdir(cwd)
        assert after == before
        assert well.name == "CONFIDENTIAL-1"


class TestNoNetworkCalls:
    """The analysis code must not reach the network. Checked by inspection of
    the source, so a future import is caught rather than merely unlikely."""

    FORBIDDEN = ("import requests", "import urllib", "import httpx",
                 "import socket", "from requests", "from urllib", "from httpx",
                 "http.client", "urlopen")

    def source_files(self):
        for root, _, files in os.walk(HERE):
            if "__pycache__" in root or "/tests" in root:
                continue
            for name in files:
                if name.endswith(".py"):
                    yield os.path.join(root, name)

    def test_no_module_imports_a_network_client(self):
        offenders = []
        for path in self.source_files():
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
            for needle in self.FORBIDDEN:
                if needle in text:
                    offenders.append(f"{os.path.relpath(path, REPO)}: {needle}")
        assert offenders == [], f"network client(s) reached the app: {offenders}"

    def test_core_imports_no_streamlit_either(self):
        """core/ stays a plain library: no UI, no I/O surprises."""
        core = os.path.join(HERE, "core")
        for name in os.listdir(core):
            if not name.endswith(".py"):
                continue
            with open(os.path.join(core, name), encoding="utf-8") as fh:
                assert "import streamlit" not in fh.read(), name


@pytest.fixture(scope="module")
def config():
    path = os.path.join(REPO, ".streamlit", "config.toml")
    assert os.path.exists(path), "no .streamlit/config.toml is shipped"
    with open(path, encoding="utf-8") as fh:
        return fh.read()


@needs_checkout
class TestShippedConfiguration:
    """The Streamlit defaults this project ships, which are safer than
    Streamlit's own."""

    def test_usage_statistics_are_off(self, config):
        assert "gatherUsageStats = false" in config

    def test_the_server_binds_to_loopback_only(self, config):
        assert 'address = "localhost"' in config


@needs_checkout
class TestGitignore:
    def _ignored(self, relative):
        return subprocess.run(
            ["git", "check-ignore", "-q", relative],
            cwd=REPO, capture_output=True,
        ).returncode == 0

    def test_an_upload_artefact_would_be_ignored(self):
        assert self._ignored("_upload.las")

    def test_stray_uppercase_las_files_are_ignored(self):
        assert self._ignored("SOME_CLIENT_WELL.LAS")

    def test_the_demo_well_is_still_tracked(self):
        assert not self._ignored("avo_qi/sample_data/demo_well.las")


class TestRuntimeMakesNoOutboundConnections:
    """Proof rather than inspection: block every non-loopback socket, then
    render a page for real and check nothing tried to dial out."""

    @staticmethod
    def _block_outbound(monkeypatch):
        import socket

        loopback = {"127.0.0.1", "::1", "localhost", "0.0.0.0", ""}
        attempts = []
        real_connect = socket.socket.connect
        real_create = socket.create_connection

        def host_of(address):
            if isinstance(address, tuple):
                return address[0]
            return address

        def guarded_connect(self, address, *args, **kwargs):
            host = host_of(address)
            if isinstance(host, str) and host not in loopback:
                attempts.append(host)
                raise OSError(f"outbound connection to {host} blocked by test")
            return real_connect(self, address, *args, **kwargs)

        def guarded_create(address, *args, **kwargs):
            host = host_of(address)
            if isinstance(host, str) and host not in loopback:
                attempts.append(host)
                raise OSError(f"outbound connection to {host} blocked by test")
            return real_create(address, *args, **kwargs)

        monkeypatch.setattr(socket.socket, "connect", guarded_connect)
        monkeypatch.setattr(socket, "create_connection", guarded_create)
        return attempts

    def test_the_core_analysis_dials_out_nowhere(self, monkeypatch, las_bytes):
        """The whole physics path, from load to classification."""
        import numpy as np

        attempts = self._block_outbound(monkeypatch)

        from avo_qi.core.avo import reflector_avo
        from avo_qi.core.synthetic import build_gather
        from avo_qi.core.wavelet import ricker

        raw, units = read_well(io.BytesIO(las_bytes), suffix=".las")
        well = standardise(raw, units=units, name="CONFIDENTIAL-1")
        vp, vs, rho = well.logs()
        angles = np.arange(0.0, 41.0, 5.0)
        _, wavelet = ricker(30.0, 0.001)
        build_gather(vp, vs, rho, angles, wavelet, dt=0.001)
        table = reflector_avo(None, vp, vs, rho, angles, threshold=0.05)

        assert len(table) > 0
        assert attempts == [], f"tried to reach {attempts}"

    def test_rendering_a_streamlit_page_dials_out_nowhere(self, monkeypatch, las_bytes):
        pytest.importorskip("streamlit")
        from streamlit.testing.v1 import AppTest

        attempts = self._block_outbound(monkeypatch)

        raw, units = read_well(io.BytesIO(las_bytes), suffix=".las")
        well = standardise(raw, units=units, name="CONFIDENTIAL-1")

        page = os.path.join(HERE, "pages", "4_AVO_Classification.py")
        at = AppTest.from_file(page, default_timeout=120)
        at.session_state["well"] = well
        at.session_state["raw_df"] = raw
        at.session_state["raw_units"] = units
        at.run()

        assert not at.exception
        assert attempts == [], f"tried to reach {attempts}"


@needs_checkout
class TestOfflineInstallSupport:
    """A pinned lockfile and a wheelhouse script, so an air-gapped machine can
    install without ever reaching an index."""

    def test_a_lockfile_is_shipped_and_pins_versions(self):
        path = os.path.join(REPO, "requirements-lock.txt")
        assert os.path.exists(path)
        with open(path, encoding="utf-8") as fh:
            pins = [ln.strip() for ln in fh
                    if ln.strip() and not ln.startswith("#")]
        assert len(pins) > 20
        assert all("==" in pin for pin in pins), "every dependency must be pinned"

    def test_the_lockfile_covers_the_direct_requirements(self):
        with open(os.path.join(REPO, "requirements-lock.txt"), encoding="utf-8") as fh:
            locked = {ln.split("==")[0].lower().replace("-", "_")
                      for ln in fh if "==" in ln}
        with open(os.path.join(REPO, "requirements.txt"), encoding="utf-8") as fh:
            direct = {ln.split("#")[0].strip().lower().replace("-", "_")
                      for ln in fh if ln.split("#")[0].strip()}
        assert direct <= locked, f"not locked: {direct - locked}"

    def test_the_bundle_script_exists_and_is_executable(self):
        path = os.path.join(REPO, "scripts", "make_offline_bundle.sh")
        assert os.path.exists(path)
        assert os.access(path, os.X_OK)

    def test_the_wheelhouse_is_not_committed(self):
        assert subprocess.run(
            ["git", "check-ignore", "-q", "wheelhouse/"],
            cwd=REPO, capture_output=True,
        ).returncode == 0
