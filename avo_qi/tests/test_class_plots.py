"""Where the classes are: the plainest plot on the page.

Everything else on the AVO page reduces the reflectors to a number — a slope,
an effect size, a rank. This one just shows them where they are, which makes
its correctness entirely a matter of layout: the right depth axis, pointing
the right way, with the class carried by something other than colour alone.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

streamlit = pytest.importorskip("streamlit")
pytest.importorskip("plotly")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from avo_qi.analysis import reflector_analysis            # noqa: E402
from avo_qi.ui import (CLASS_COLOURS, Settings, class_depth_axis,  # noqa: E402
                       class_depth_figure, class_panel_options)


@pytest.fixture(scope="module")
def table():
    from test_app_smoke import demo_well

    return reflector_analysis(demo_well()[0], Settings())["table"]


def marks(figure):
    """The data traces, leaving out the legend-only placeholder entries."""
    return [d for d in figure.data
            if d.x is not None and len(d.x) and d.x[0] is not None]


class TestTheDepthAxis:
    def test_it_points_downwards(self, table):
        """Depth increases down a well and must increase down the plot. A
        reversed axis is the whole reason this is not a default scatter."""
        figure = class_depth_figure(table, columns=["A_shuey"])
        assert figure.layout.yaxis.autorange == "reversed"

    def test_a_true_vertical_reference_wins_over_measured_depth(self, table):
        """MD starts at a rig floor, so it says where a class is in the *hole*.
        Given a subsea datum the plot has to prefer it."""
        assert class_depth_axis(table)[0] == "depth"
        with_datum = table.assign(tvdss=table["depth"] - 30.0)
        assert class_depth_axis(with_datum)[0] == "tvdss"

    def test_every_panel_shares_it(self, table):
        """The claim the caption makes: a horizontal line across the figure is
        one reflector. That is only true if the panels share the y axis."""
        figure = class_depth_figure(table, columns=["A_shuey", "phi_res"])
        for axis in ("yaxis2", "yaxis3"):
            assert getattr(figure.layout, axis).matches == "y"

    def test_a_table_with_no_depth_at_all_draws_nothing(self):
        import pandas as pd

        bare = pd.DataFrame({"avo_class": ["III"], "A_shuey": [-0.1]})
        assert class_depth_figure(bare) is None


class TestIdentityIsNotColourAlone:
    def test_the_class_has_an_axis_of_its_own(self, table):
        """The first panel puts the class on the x axis, so the figure works
        printed and for a reader who cannot separate the reds from the
        greens — the palette warns on contrast, which obliges exactly this."""
        figure = class_depth_figure(table, columns=["A_shuey"])
        ticks = list(figure.layout.xaxis.ticktext)
        assert len(ticks) == len(CLASS_COLOURS)
        # Abbreviated only where the full name would rotate into a wedge of
        # empty space; the legend still carries it in full.
        assert ticks[:-1] == list(CLASS_COLOURS)[:-1]
        assert ticks[-1] == "bg"
        assert "background/other" in [d.name for d in figure.data]

    def test_the_class_order_is_fixed_not_sorted_by_count(self, table):
        """So the same well looks the same on every rerun, and two wells look
        like each other."""
        few = table[table["avo_class"].isin(["IV", "III"])]
        figure = class_depth_figure(few, columns=["A_shuey"])
        assert list(figure.layout.xaxis.ticktext) == \
            list(class_depth_figure(table, columns=["A_shuey"])
                 .layout.xaxis.ticktext)

    def test_each_class_keeps_its_own_colour(self, table):
        figure = class_depth_figure(table, columns=["A_shuey"])
        for trace in marks(figure):
            assert trace.marker.color == CLASS_COLOURS[trace.name]

    def test_overlapping_events_stay_countable(self, table):
        """A surface-coloured ring on every mark, so a stack of reflectors at
        one depth reads as several rather than as one blob."""
        figure = class_depth_figure(table, columns=["A_shuey"])
        for trace in marks(figure):
            assert trace.marker.line.width >= 1
            assert trace.marker.line.color == "#FFFFFF"


class TestThePanels:
    def test_the_class_panel_is_always_first(self, table):
        figure = class_depth_figure(table, columns=["A_shuey", "phi_res"])
        titles = [note.text for note in figure.layout.annotations]
        assert titles[0] == "AVO class"
        assert len(titles) == 3

    def test_it_offers_only_columns_the_well_actually_has(self, table):
        options = class_panel_options(table)
        assert "A_shuey" in options and "phi_res" in options
        assert set(options) <= set(table.columns)

        without = table.drop(columns=["phi_res"])
        assert "phi_res" not in class_panel_options(without)

    def test_a_column_of_nothing_but_nulls_is_not_offered(self, table):
        blanked = table.assign(sw_res=np.nan)
        assert "sw_res" not in class_panel_options(blanked)

    def test_no_panels_still_draws_the_classes(self, table):
        figure = class_depth_figure(table, columns=[])
        assert figure is not None
        assert [note.text for note in figure.layout.annotations] == ["AVO class"]

    def test_the_demo_wells_gas_sand_sits_where_the_well_puts_it(self, table):
        """A known answer rather than a shape check: the plot has to place the
        gas-sand top at that reflector's own depth.

        Note the well has *two* shale-over-sand tops — the gas sand and the
        cemented streak below it — and only one is Class III, which is why the
        pair alone does not identify it.
        """
        figure = class_depth_figure(table, columns=["A_shuey"])
        expected = table.loc[(table["litho_pair"] == "shale over sand")
                             & (table["avo_class"] == "III"),
                             "depth"].to_numpy(float)
        assert expected.size == 1
        drawn = np.concatenate([np.asarray(t.y, float) for t in marks(figure)
                                if t.name == "III"])
        assert set(np.round(expected, 3)) <= set(np.round(drawn, 3))
        # And it is not drawn under any other class.
        for other in ("I", "IV"):
            elsewhere = np.concatenate(
                [np.asarray(t.y, float) for t in marks(figure)
                 if t.name == other] or [np.array([])])
            assert not set(np.round(expected, 3)) & set(np.round(elsewhere, 3))


class TestSeveralWells:
    @staticmethod
    def two_wells(table):
        import pandas as pd
        from test_multi_well import second_well

        other = reflector_analysis(second_well()[0], Settings())["table"]
        return pd.concat([table.assign(well="DEMO-1"),
                          other.assign(well="15/9-19-A")], ignore_index=True)

    def test_the_well_is_the_shape_and_the_class_is_the_colour(self, table):
        """Two encodings, two variables. Using colour for both would make a
        Class III in one well look like a Class I in another."""
        figure = class_depth_figure(self.two_wells(table), columns=["A_shuey"],
                                    depth_column="depth", split_column="well")
        shapes = {t.marker.symbol for t in marks(figure)}
        assert len(shapes) == 2
        for trace in marks(figure):
            assert trace.marker.color == CLASS_COLOURS[trace.name]

    def test_each_class_appears_once_in_the_legend(self, table):
        """Not once per well per panel, which is how a legend gets forty
        entries and stops being a legend."""
        figure = class_depth_figure(self.two_wells(table),
                                    columns=["A_shuey", "phi_res"],
                                    depth_column="depth", split_column="well")
        shown = [d.name for d in figure.data if d.showlegend]
        assert len(shown) == len(set(shown))
        assert {"DEMO-1", "15/9-19-A"} <= set(shown)

    def test_the_class_legend_survives_a_well_with_nothing_on_this_axis(self):
        """The bug the browser caught. Keying the class legend to the *first*
        well loses it entirely when that well has no TVDSS — which is the
        common case, because a well with no datum is exactly the one that
        cannot be drawn on a subsea axis."""
        import pandas as pd
        from test_app_smoke import demo_well
        from test_multi_well import second_well

        no_datum = reflector_analysis(demo_well()[0], Settings())["table"]
        with_datum = reflector_analysis(second_well()[0], Settings())["table"]
        both = pd.concat([no_datum.assign(well="DEMO-1"),
                          with_datum.assign(well="15/9-19-A",
                                            tvdss=with_datum["depth"] - 30.0)],
                         ignore_index=True)

        figure = class_depth_figure(both, columns=["A_shuey"],
                                    split_column="well")
        shown = [d.name for d in figure.data if d.showlegend]
        assert set(with_datum["avo_class"]) <= set(shown)

    def test_a_well_that_cannot_be_drawn_gets_no_legend_entry(self):
        """A legend entry for a well with no points reads as 'it is in here
        somewhere' rather than 'it could not be drawn'. It is named in a
        caption instead."""
        import pandas as pd
        from test_app_smoke import demo_well
        from test_multi_well import second_well

        no_datum = reflector_analysis(demo_well()[0], Settings())["table"]
        with_datum = reflector_analysis(second_well()[0], Settings())["table"]
        both = pd.concat([no_datum.assign(well="DEMO-1"),
                          with_datum.assign(well="15/9-19-A",
                                            tvdss=with_datum["depth"] - 30.0)],
                         ignore_index=True)

        figure = class_depth_figure(both, columns=["A_shuey"],
                                    split_column="well")
        shown = [d.name for d in figure.data if d.showlegend]
        assert "DEMO-1" not in shown
        assert "15/9-19-A" in shown
        assert figure._avo_qi_missing == ["DEMO-1"]
