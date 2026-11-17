"""Shared presentation for all five runnable checkpoints."""

from pathlib import Path

import streamlit as st

BASE = Path(__file__).resolve().parent
STEPS = (
    "A redesigned workspace",
    "Upload and store documents",
    "Rename and delete documents",
    "Extract readable text",
    "Create document chunks",
)


def chat_exchange(prompt, answer):
    return [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]


def show_transcript(messages):
    for message in messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])


def run_workshop(step):
    from memory import persistent_chat
    from saved_memories import show_saved_memories
    from selective_memory import select_saved_facts

    st.set_page_config(page_title="Copilot Studio", page_icon="✦", layout="wide")
    st.html((BASE / "theme.css").read_text())
    st.sidebar.markdown("### ✦ Copilot Studio")
    st.sidebar.caption("YOUR LOCAL LEARNING WORKSPACE")
    st.sidebar.divider()

    def home():
        st.html("""
        <section class="welcome">
          <p class="eyebrow">YOUR SPACE TO THINK, LEARN, AND BUILD</p>
          <h1>Good ideas start<br>with a conversation.</h1>
          <p class="intro">Meet your local AI copilot. Bring a question, keep the
          details that matter, and prepare your documents for smarter answers.</p>
        </section>
        """)
        a, b, _ = st.columns([1, 1, 2])
        if a.button("Start a conversation", type="primary", use_container_width=True):
            st.switch_page(chat_page)
        st.divider()
        for column, title, body in zip(
            st.columns(3),
            ("01 / Your conversations", "02 / Your memory", "03 / Your documents"),
            (
                "A focused place to explore questions with a model running on your computer.",
                "Keep recent context and choose the details you want to save for later.",
                "Upload materials and inspect how text becomes smaller, searchable pieces.",
            ),
        ):
            with column:
                st.subheader(title)
                st.write(body)
        st.caption(f"NOVEMBER 18 WORKSHOP · CHECKPOINT {step} OF 5 · {STEPS[step - 1]}")

    def chat():
        persistent_chat(BASE / "data" / "chat_memory.json", show_saved_memories, select_saved_facts)

    (BASE / "data").mkdir(exist_ok=True)
    home_page = st.Page(home, title="Welcome", icon=":material/home:", default=True)
    chat_page = st.Page(chat, title="Conversations", icon=":material/chat_bubble:", url_path="chat")
    pages = [home_page, chat_page]
    st.navigation(pages).run()
