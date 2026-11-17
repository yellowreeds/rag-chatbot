"""Checkpoint 4: save, replace, and delete named facts."""

from dataclasses import replace

import streamlit as st

from memory import Memory

FACT_LABELS = {
    "project_framework": "Project framework",
    "answer_style": "Preferred answer style",
}


def show_saved_memories(memory, save_change):
    with st.sidebar.expander("Saved memories", expanded=True):
        st.caption("Replacing or deleting a fact also resets the conversation so old turns cannot supply it.")
        for key, label in FACT_LABELS.items():
            current = memory.facts.get(key, "")
            st.write(f"**{label}**")
            st.text(current or "(empty)")
            with st.form(f"edit_{key}", clear_on_submit=True):
                value = st.text_input(f"New value for {label.lower()}", max_chars=60).strip()
                save = st.form_submit_button("Save value")
                delete = st.form_submit_button("Delete value", disabled=not current)

            if not (save or delete):
                continue
            if save and not value:
                st.warning("Enter a value to save, or use Delete value to clear it.")
                continue
            if save and value == current:
                continue

            facts = dict(memory.facts)
            if delete:
                del facts[key]
            else:
                facts[key] = value
            updated = Memory(facts=facts) if current else replace(memory, facts=facts)
            save_change(updated, clear_chat=bool(current))

