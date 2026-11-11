"""Checkpoint 1: landing page, chat screen, and echo."""

import streamlit as st


WELCOME_HERO = """
<style>
    .copilot-welcome {
        background: linear-gradient(135deg, #edf5ff, #dcecff);
        border: 1px solid #ccdff6;
        border-radius: 1rem;
        padding: clamp(1.5rem, 5vw, 3rem);
        color: #142a49;
    }
    .copilot-welcome h1 {
        color: inherit;
        font-size: clamp(2rem, 5vw, 3.25rem);
        line-height: 1.1;
        padding: 0;
        margin: 0 0 1rem;
    }
    .copilot-welcome p:last-child {
        color: #314b6c;
        max-width: 52ch;
        margin: 0;
    }
</style>
<section class="copilot-welcome" aria-labelledby="copilot-welcome-title">
    <h1 id="copilot-welcome-title">Welcome to Local AI Copilot</h1>
    <p>Take your terminal chatbot to the browser. Build a chat interface,
       connect it to a local model, and teach it what to remember.</p>
</section>
"""


def show_transcript(messages):
    for message in messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])


def chat_exchange(prompt, answer):
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer},
    ]


def echo_chat():
    st.title("Chat")
    st.caption("This first version echoes your message.")
    transcript = st.session_state.setdefault("transcript", [])
    prompt = (st.chat_input("Type a message") or "").strip()
    if prompt:
        transcript.extend(chat_exchange(prompt, f"You said: {prompt}"))
    show_transcript(transcript)


def run_app(step, description, chat):
    st.set_page_config(page_title=f"RAG Class · Step {step}")

    def landing_page():
        st.html(WELCOME_HERO)
        if st.button("Open chatbot →", type="primary"):
            st.switch_page(chat_page)

        with st.container(border=True):
            st.subheader(f"Checkpoint {step} of 5")
            st.write(description)
            st.progress(step / 5)

    def chatbot_page():
        if st.button("← Back to home"):
            st.switch_page(home_page)
        chat()

    home_page = st.Page(landing_page, title="Welcome", default=True)
    chat_page = st.Page(chatbot_page, title="Chatbot", url_path="chat")
    st.navigation([home_page, chat_page], position="hidden").run()


if __name__ == "__main__":
    run_app(1, "Build a landing page and a chat screen that echoes your input.", echo_chat)
