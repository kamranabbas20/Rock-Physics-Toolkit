"""The self-contained HTML report."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from avo_qi.report import (
    Figures,
    PanelReport,
    Section,
    frame_html,
    key_values_html,
    no_external_references,
    paragraph_html,
    report_html,
)


class TestSelfContained:
    """The report's whole promise: it opens on a laptop with no network, six
    months from now, whatever happened to any CDN in between."""

    def test_a_remote_script_is_rejected(self):
        for markup in (
            '<script src="https://cdn.plot.ly/plotly.min.js"></script>',
            "<script src='http://example.com/x.js'></script>",
            '<script src="//cdn.example.com/x.js"></script>',      # protocol-relative
            '<script type="text/javascript" src="https://x/y.js"></script>',
        ):
            assert not no_external_references(markup), markup

    def test_remote_styles_and_images_are_rejected(self):
        assert not no_external_references(
            '<link rel="stylesheet" href="https://x.com/a.css">')
        assert not no_external_references('<img src="http://x.com/a.png">')
        assert not no_external_references('<a href="//x.com/page">x</a>')

    def test_a_url_inside_an_inline_script_is_not_a_reference(self):
        """The subtlety that made a correct report look broken.

        Plotly's bundle is inlined — that is the point — and its map support
        carries OpenStreetMap and MapLibre attribution links and an unpkg icon
        URL in its source. Nothing in a report draws a map, so none of it is
        ever fetched. A scan that does not exclude script bodies calls a
        perfectly self-contained file external.
        """
        bundle = ('<script>var attribution = '
                  '"<a href=\\"https://www.openstreetmap.org/copyright\\">OSM</a>";'
                  ' var icons = "https://unpkg.com/maki@2.1.0/icons/";</script>')
        assert no_external_references(bundle + "<p>report</p>")
        # ...but a real remote script alongside it is still caught.
        assert not no_external_references(
            bundle + '<script src="https://cdn.example.com/x.js"></script>')

    def test_a_plain_document_passes(self):
        assert no_external_references("")
        assert no_external_references(None)
        assert no_external_references(
            report_html("t", [Section("s", ["<p>x</p>"])], generated="2026-01-01"))


class TestFrameHtml:
    def test_numbers_are_right_aligned_and_text_is_not(self):
        frame = pd.DataFrame({"name": ["a", "b"], "value": [1.5, 2.5]})
        out = frame_html(frame)
        assert '<th class="num">value</th>' in out
        assert "<th>name</th>" in out
        assert '<td class="num">1.5</td>' in out

    def test_non_finite_numbers_render_blank_rather_than_nan(self):
        frame = pd.DataFrame({"v": [1.0, np.nan, np.inf]})
        out = frame_html(frame)
        assert "nan" not in out.lower()
        assert "inf" not in out.lower()

    def test_booleans_are_not_treated_as_numbers(self):
        frame = pd.DataFrame({"flag": [True, False]})
        out = frame_html(frame)
        assert "<th>flag</th>" in out
        assert "<td>True</td>" in out

    def test_it_truncates_and_says_so(self):
        frame = pd.DataFrame({"v": range(50)})
        out = frame_html(frame, max_rows=10)
        assert "Showing 10 of 50 rows" in out
        assert out.count("<tr>") == 11          # header plus ten

    def test_an_empty_frame_is_not_an_error(self):
        assert "Nothing to show" in frame_html(pd.DataFrame())
        assert "Nothing to show" in frame_html(None)

    def test_content_is_escaped(self):
        frame = pd.DataFrame({"name": ["<script>alert(1)</script>"]})
        out = frame_html(frame)
        assert "<script>" not in out
        assert "&lt;script&gt;" in out


class TestAssembly:
    def test_the_document_is_well_formed_enough_to_open(self):
        doc = report_html("Title", [Section("A", ["<p>one</p>"], lead="lead")],
                          subtitle="sub", generated="2026-01-01 09:00")
        assert doc.startswith("<!doctype html>")
        assert doc.rstrip().endswith("</html>")
        assert doc.count("<html") == doc.count("</html>") == 1
        assert doc.count("<body") == doc.count("</body>") == 1
        for probe in ("Title", "sub", "2026-01-01 09:00", "<h2>A</h2>",
                      "lead", "<p>one</p>"):
            assert probe in doc

    def test_the_timestamp_is_injectable_so_output_is_reproducible(self):
        a = report_html("T", [], generated="2026-01-01 09:00")
        b = report_html("T", [], generated="2026-01-01 09:00")
        assert a == b

    def test_titles_are_escaped(self):
        doc = report_html("<script>x</script>", [], generated="x")
        assert "<title><script>" not in doc
        assert "&lt;script&gt;" in doc

    def test_empty_blocks_are_dropped(self):
        assert Section("A", ["", None, "<p>keep</p>"]).render().count("<p>") == 1

    def test_key_values_escapes_both_sides(self):
        out = key_values_html([("<k>", "<v>")])
        assert "&lt;k&gt;" in out and "&lt;v&gt;" in out

    def test_paragraph_kinds(self):
        assert paragraph_html("x") == "<p>x</p>"
        assert 'class="note"' in paragraph_html("x", kind="note")
        assert "warn" in paragraph_html("x", kind="warn")


class TestFigureEmbedding:
    """Plotly is inlined once: repeating a multi-megabyte bundle per figure
    would multiply the file size by the figure count."""

    @staticmethod
    def _figure():
        import plotly.graph_objects as go

        return go.Figure(go.Scatter(x=[1, 2, 3], y=[1, 4, 9]))

    def test_only_the_first_figure_carries_the_library(self):
        pytest.importorskip("plotly")
        from avo_qi.report import figure_html

        with_bundle = figure_html(self._figure(), first=True)
        without = figure_html(self._figure())
        assert len(with_bundle) > 1_000_000     # the bundle is in there
        assert len(without) < 100_000           # and not in there
        assert "plotly" in without.lower()      # still a real figure

    def test_a_missing_figure_is_not_an_error(self):
        from avo_qi.report import figure_html

        assert figure_html(None) == ""


def _figure(y=(1, 2, 3)):
    """A figure with as little in it as a figure can have."""
    go = pytest.importorskip("plotly.graph_objects")
    return go.Figure(go.Scatter(y=list(y)))


class TestTheFigureAllocator:
    """The bundle has to be inlined exactly once per document.

    Twice and the file grows by several megabytes for nothing; not at all and
    it draws nothing offline, which is the one thing the report promises. With
    a dozen sections — several of them conditional on what the well carries —
    *which* figure comes first is not something a caller can be asked to know.
    """

    def test_only_the_first_figure_carries_the_bundle(self):
        emit = Figures()
        first, second, third = (emit(_figure()) for _ in range(3))
        assert "Plotly.newPlot" in first and "Plotly.newPlot" in second
        # The bundle is what makes the first block enormous.
        assert len(first) > 10 * len(second)
        assert len(third) < len(first)

    def test_a_section_that_draws_nothing_does_not_consume_it(self):
        """The reason this is an allocator and not a counter: on a well with
        no petrophysics the first two sections drop out entirely, and the
        bundle has to move to whichever figure actually gets drawn."""
        emit = Figures()
        assert emit(None) == ""
        assert emit.first is True
        assert len(emit(_figure())) > 100_000
        assert emit.first is False

    def test_the_height_still_gets_through(self):
        assert '"height":333' in Figures()(_figure(), height=333).replace(" ", "")


class TestPanelReport:
    """A panel hands the report what it drew, rather than the report
    rebuilding it from defaults and quietly disagreeing with the screen."""

    def test_blocks_keep_the_order_they_were_added_in(self):
        panel = (PanelReport()
                 .note("first")
                 .frame(pd.DataFrame({"a": [1]}))
                 .note("last"))
        html = panel.section("T", Figures()).render()
        assert html.index("first") < html.index("<table") < html.index("last")

    def test_an_empty_panel_makes_no_section(self):
        """A heading with nothing under it reads as a failure. On a well with
        no petrophysics that is the honest outcome for half of these."""
        assert PanelReport().section("Nothing", Figures()) is None

    def test_nothing_is_added_for_an_empty_frame_or_a_missing_figure(self):
        panel = (PanelReport()
                 .figure(None)
                 .frame(None)
                 .frame(pd.DataFrame({"a": []})))
        assert panel.section("T", Figures()) is None

    def test_a_warn_note_is_called_out(self):
        rendered = PanelReport().note("careful", kind="warn").section(
            "T", Figures()).render()
        assert 'class="note warn"' in rendered

    def test_the_lead_can_come_from_the_panel_or_the_caller(self):
        panel = PanelReport(lead="from the panel").note("x")
        assert "from the panel" in panel.section("T", Figures()).render()
        assert "from the caller" in panel.section(
            "T", Figures(), lead="from the caller").render()

    def test_a_frame_caption_precedes_its_table(self):
        rendered = PanelReport().frame(
            pd.DataFrame({"a": [1]}), caption="what this is").section(
            "T", Figures()).render()
        assert rendered.index("what this is") < rendered.index("<table")
