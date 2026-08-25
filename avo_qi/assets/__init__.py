"""Brand assets, kept as SVG so they scale and stay text in the repository.

Read them through :func:`avo_qi.ui.brand`, which returns the markup rather than
a path: ``st.image`` resolves relative paths against the working directory, and
the app is run from the repository root but tested from elsewhere.
"""
