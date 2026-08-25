"""Colour constants shared by the Streamlit app and the browser-free CLI.

Kept apart from ``ui.py`` so the CLI can import them without pulling in
Streamlit or Plotly.
"""

CLASS_COLOURS = {
    "I": "#1f77b4",
    "IIp": "#17becf",
    "IIn": "#9467bd",
    "III": "#d62728",
    "IV": "#ff7f0e",
    "background/other": "#9e9e9e",
}

LITHOLOGY_COLOURS = {
    "sand": "#f0c419",
    "silty sand": "#d99a3a",
    "silt": "#8c8060",
    "shale": "#5b6b73",
    "undefined": "#c9c9c9",
}

CASE_COLOURS = {
    "in situ": "#333333",
    "brine": "#1f77b4",
    "oil": "#2ca02c",
    "gas": "#d62728",
}

#: Qualitative colours for wells in a cross-well comparison. Deliberately
#: distinct from CLASS_COLOURS and CASE_COLOURS: on a multi-well crossplot the
#: colour means *which well*, and reusing a class colour there would say
#: "Class III" to anyone reading it out of the corner of an eye.
WELL_COLOURS = [
    "#0E5A6B",   # the toolkit's own petrol, for the first well
    "#B4622D",
    "#4C6E31",
    "#7B4F9D",
    "#2E6DA4",
    "#9A2E3F",
    "#5C6B73",
    "#8A6D1F",
]


def well_colour(index):
    """Colour for the *n*-th well, wrapping round if there are many."""
    return WELL_COLOURS[int(index) % len(WELL_COLOURS)]
