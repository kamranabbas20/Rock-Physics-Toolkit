#!/usr/bin/env python3
"""Prove, on your own machine and with your own well, that nothing leaves it.

Starts the app, drives a real browser through every page — uploading a well if
you give it one — and records every network request the browser makes.  Any
request to a host other than localhost is reported as a leak and the script
exits non-zero.

    pip install playwright && playwright install chromium
    python scripts/verify_no_network.py                     # demo well
    python scripts/verify_no_network.py path/to/your.las    # your own well

If Playwright's bundled browser is not available, point it at any Chromium:

    python scripts/verify_no_network.py --chromium /path/to/chrome

This checks the browser side.  The Python side is covered by the test suite:
``pytest avo_qi/tests/test_data_handling.py`` blocks every non-loopback socket
and then runs the analysis and renders a page.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from urllib.parse import urlparse

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
PAGES = [
    ("Data and Crossplots", 9000),
    ("Synthetic Gather", 11000),
    ("AVO Classification", 13000),
    ("Rock Physics", 10000),
]


def wait_idle(page, limit=240):
    """Wait for Streamlit to stop running the script."""
    for _ in range(limit):
        if page.locator('[data-testid="stStatusWidget"]').count() == 0:
            return
        time.sleep(0.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("well", nargs="?", help="a LAS to upload; omit to use the demo well")
    parser.add_argument("--port", type=int, default=8799)
    parser.add_argument(
        "--chromium", default=os.environ.get("CHROMIUM_PATH"),
        help="path to a Chromium binary, if Playwright's own is not installed",
    )
    args = parser.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        sys.exit("playwright is needed: pip install playwright && playwright install chromium")

    url = f"http://localhost:{args.port}"
    server = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", os.path.join("avo_qi", "app.py"),
         "--server.headless", "true", "--server.port", str(args.port)],
        cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    seen, external = [], []
    try:
        time.sleep(12)
        with sync_playwright() as pw:
            launch = {"executable_path": args.chromium} if args.chromium else {}
            browser = pw.chromium.launch(**launch)
            context = browser.new_context()

            def on_request(request):
                seen.append(request.url)
                host = urlparse(request.url).hostname or ""
                if host and host not in LOCAL_HOSTS:
                    external.append(f"{request.method} {request.url}")

            context.on("request", on_request)
            page = context.new_page()
            page.goto(url, wait_until="networkidle", timeout=90000)
            page.wait_for_timeout(4000)

            page.get_by_role("link", name="Load and QC").click()
            page.wait_for_timeout(5000)
            wait_idle(page)
            if args.well:
                print(f"Uploading {args.well} …")
                page.locator('input[type="file"]').first.set_input_files(args.well)
                page.wait_for_timeout(12000)
            else:
                page.get_by_role("button", name="Load demo well").last.click()
                page.wait_for_timeout(6000)
            wait_idle(page)

            for name, wait in PAGES:
                print(f"Visiting {name} …")
                page.get_by_role("link", name=name).click()
                page.wait_for_timeout(wait)
                wait_idle(page)
                page.wait_for_timeout(2000)

            browser.close()
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()

    print(f"\nRequests captured : {len(seen)}")
    print(f"Left this machine : {len(external)}")
    if external:
        for line in external[:40]:
            print("  LEAK:", line)
        sys.exit(1)
    print("\nNothing left this machine. Every request stayed on localhost.")


if __name__ == "__main__":
    main()
