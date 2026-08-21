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
