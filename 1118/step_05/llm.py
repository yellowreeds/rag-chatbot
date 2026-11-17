"""Shared local model calls and bounded retries."""

import ollama
import streamlit as st

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

