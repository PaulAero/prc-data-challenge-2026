"""Data preparation and analysis helpers behind the local data explorer.

Everything here is pure polars: functions take and return (Lazy)DataFrames so
they can be tested without Streamlit. The Streamlit app lives in
``apps/explorer/app.py`` and the cache builder in
``scripts/explore/build_explorer_cache.py``.
"""
