"""The toolkit's identity: the mark, the lockup, and the theme.

Two properties are worth holding rather than trusting. The assets must fetch
nothing, because the app and its report both promise to work with no network at
all. And the accent colour must not be a colour that already *means* something
in the plots — Streamlit's default primary is a red in the same family as
Class III and as the negative fill on every trace, which makes a button look
like a finding.
"""

from __future__ import annotations

import os
import tomllib
import xml.etree.ElementTree as ET

import pytest

from avo_qi.ui_colours import CASE_COLOURS, CLASS_COLOURS, LITHOLOGY_COLOURS

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASSETS = os.path.join(ROOT, "avo_qi", "assets")
CONFIG = os.path.join(ROOT, ".streamlit", "config.toml")

MARK = "mark.svg"
LOGO = "logo.svg"


def read(name):
    with open(os.path.join(ASSETS, name), "r", encoding="utf-8") as handle:
        return handle.read()


class TestTheAssets:
    @pytest.mark.parametrize("name", [MARK, LOGO])
    def test_each_one_is_well_formed_svg(self, name):
        markup = read(name)
        root = ET.fromstring(markup)
        assert root.tag.endswith("svg")
        assert root.get("viewBox")

    @pytest.mark.parametrize("name", [MARK, LOGO])
    def test_nothing_is_fetched(self, name):
        """The offline promise, held for the brand as it is for the report.

        The SVG namespace declaration is a URI and not a fetch — it names the
        dialect and is never dereferenced — so it is excluded, the same way
        ``report.no_external_references`` excludes the URLs written inside the
        Plotly bundle it inlines.
        """
        markup = read(name).replace('xmlns="http://www.w3.org/2000/svg"', "")
        assert "http://" not in markup and "https://" not in markup
        assert "@import" not in markup and "url(" not in markup

    @pytest.mark.parametrize("name", [MARK, LOGO])
    def test_it_scales(self, name):
        """No fixed width or height: the same file has to serve a 16 px favicon
        and a 280 px lockup."""
        root = ET.fromstring(read(name))
        assert root.get("width") is None and root.get("height") is None

    def test_the_mark_uses_the_traces_own_colours(self):
        """The mark is a variable-area trace, so it must be filled with the two
        colours the AVO page fills a trace with. Anything else and the logo
        drifts from the thing it depicts."""
        from avo_qi.ui import TRACE_FILL_NEGATIVE, TRACE_FILL_POSITIVE

        def opaque(rgba):
            red, green, blue = (int(v) for v in
                                rgba.split("(")[1].split(")")[0].split(",")[:3])
            return f"#{red:02X}{green:02X}{blue:02X}"

        markup = read(MARK)
        assert opaque(TRACE_FILL_NEGATIVE) in markup
        assert opaque(TRACE_FILL_POSITIVE) in markup

    def test_the_lockup_carries_the_mark_and_the_name(self):
        markup = read(LOGO)
        assert "AVO" in markup and "QI" in markup
        assert "WELL ANALYSIS TOOLKIT" in markup
        # The mark is embedded rather than referenced, so the lockup is one file.
        assert markup.count("<path") >= 6

    def test_brand_returns_markup_not_a_path(self):
        """``st.image`` resolves a relative path against the working directory,
        and the tests do not run from the repository root."""
        from avo_qi.ui import brand

        assert brand(MARK).lstrip().startswith("<svg")
        assert brand(LOGO).lstrip().startswith("<svg")


class TestTheTheme:
    @staticmethod
    def theme():
        with open(CONFIG, "rb") as handle:
            return tomllib.load(handle)["theme"]

    def test_the_accent_is_set_and_light_is_pinned(self):
        theme = self.theme()
        assert theme["primaryColor"].upper() == "#0E5A6B"
        # Every Plotly figure is drawn on a light template; dark chrome around
        # white plots reads as a mistake rather than a choice.
        assert theme["base"] == "light"

    def test_the_accent_means_nothing_in_the_plots(self):
        """The reason for moving off Streamlit's red: an accent that is also a
        class colour makes a button look like a finding."""
        accent = self.theme()["primaryColor"].upper()
        semantic = {c.upper() for c in
                    list(CLASS_COLOURS.values()) + list(CASE_COLOURS.values())
                    + list(LITHOLOGY_COLOURS.values())}
        assert accent not in semantic

    def test_the_app_and_the_assets_agree_on_the_accent(self):
        from avo_qi.ui import BRAND_PETROL

        assert BRAND_PETROL.upper() == self.theme()["primaryColor"].upper()
        assert BRAND_PETROL.upper() in read(MARK).upper()

    def test_the_fonts_are_system_stacks(self):
        """A webfont would be an outbound request on every page load."""
        theme = self.theme()
        for key in ("font", "headingFont", "codeFont"):
            assert "http" not in theme[key]
            assert "," in theme[key]        # a stack, not a single named face
