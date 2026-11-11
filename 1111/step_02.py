"""Checkpoint 2: replace the echo with a local Ollama reply."""

import ollama
import streamlit as st

from step_01 import chat_exchange, run_app, show_transcript

MODEL = "llama3.1:latest"
SYSTEM_PROMPT = "Be helpful and concise."
CHAT_ERRORS = (ConnectionError, ollama.ResponseError, ollama.RequestError, ValueError)


def ask_ollama(messages, options=None, retry_tokens=None):
    response = ollama.chat(model=MODEL, messages=messages, options=options, think=False)
    if retry_tokens is not None and response.done_reason == "length":
        retry_options = {**(options or {}), "num_predict": retry_tokens}
        response = ollama.chat(model=MODEL, messages=messages, options=retry_options, think=False)
        if response.done_reason == "length":
            raise ValueError("The reply reached the output limit. Ask for a shorter answer.")
    answer = (response.message.content or "").strip()
    if not answer:
        raise ValueError("Ollama returned an empty answer.")
    return answer


def show_chat_error(error):
    st.error(f"Chat request failed: {error}")
    st.caption(f"For connection or model errors, open Ollama and check that {MODEL} is installed.")


def local_chat():
    st.title("Chat")
    st.caption("The transcript stays on screen. Ollama receives the current question only.")
    transcript = st.session_state.setdefault("transcript", [])
    prompt = (st.chat_input("Ask a question") or "").strip()
    if prompt:
        request = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        try:
            with st.spinner("Waiting for Ollama…"):
                answer = ask_ollama(request)
        except CHAT_ERRORS as error:
            show_chat_error(error)
        else:
            transcript.extend(chat_exchange(prompt, answer))
    show_transcript(transcript)


if __name__ == "__main__":
    run_app(2, "Connect the chat to Ollama and try a follow-up question.", local_chat)
