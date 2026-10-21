"""View 9: hypotheses notepad, appended to docs/explorer_notes.md."""

from __future__ import annotations

import streamlit as st

import ui
from prc.explore.analysis import append_note
from prc.explore.common import NOTES_PATH


def _save() -> None:
    text = st.session_state.get("n_text", "").strip()
    if not text:
        st.session_state["n_msg"] = "Nothing to save."
        return
    append_note(NOTES_PATH, text, st.session_state.get("n_ctx", "").strip())
    st.session_state["n_text"] = ""
    st.session_state["n_msg"] = f"Appended to {NOTES_PATH.relative_to(NOTES_PATH.parents[1])}."


def render(f: ui.Filters) -> None:
    st.header("9. Hypotheses notepad")
    st.markdown(f"Notes are appended with a timestamp to `{NOTES_PATH}` (local file, never sent anywhere).")
    st.text_input("Context (optional, e.g. view and airport-day)", key="n_ctx",
                  placeholder="replay EGLL 2025-07-15")
    st.text_area("Observation or hypothesis", key="n_text", height=160)
    st.button("Append note", on_click=_save, key="n_save")
    if msg := st.session_state.pop("n_msg", None):
        st.success(msg)
    st.caption("Current sidebar filters: " + f.describe())
    if NOTES_PATH.exists():
        st.subheader("Existing notes")
        st.markdown(NOTES_PATH.read_text(encoding="utf-8"))
