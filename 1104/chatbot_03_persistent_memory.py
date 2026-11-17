"""November 4, version 3: persistent summary plus recent history.

Run: python chatbot_03_persistent_memory.py
Requires the existing rag-class environment, ollama library, and gemma3:1b.

This standalone script extends version 2 by loading and saving a JSON file.
The file lives beside this script, independent of the terminal's folder.
The console conversation can therefore survive a program restart.
A browser UI can later reuse the same load/save idea when a session restarts.

For teaching, estimate 1 token per 4 characters, plus message overhead.
This is NOT the model's tokenizer; estimates vary with language and content.
The safety margin helps, but does not guarantee an exact model-token limit.
"""

import json
import math
from pathlib import Path

import ollama

from chatbot_memory import (
    ANSWER_TOKENS,
    CONTEXT_TOKENS,
    INPUT_BUDGET,
    MODEL,
    SUMMARY_TOKENS,
    estimate_tokens,
    prepare_context,
    show_memory,
)

def load_memory():
    """Restore a summary and complete exchanges, or start with no saved memory."""
    try:
        raw = MEMORY_FILE.read_text(encoding="utf-8")
    except FileNotFoundError:
        return "", []

    data = json.loads(raw)
    if not isinstance(data, dict) or data.get("version") != 1:
        raise ValueError("Unsupported memory file format.")

    summary = data.get("summary")
    messages = data.get("messages")
    if not isinstance(summary, str) or not isinstance(messages, list):
        raise ValueError("Memory needs a text summary and a messages list.")
    if len(summary) > SUMMARY_TOKENS * 4:
        raise ValueError("The saved summary exceeds this example's size limit.")
    if len(messages) % 2:
        raise ValueError("Saved history must contain complete user/assistant pairs.")

    # Restore only conversation turns; keep system instructions in the code.
    history = []
    for index, message in enumerate(messages):
        expected_role = "user" if index % 2 == 0 else "assistant"
        if (
            not isinstance(message, dict)
            or message.get("role") != expected_role
            or not isinstance(message.get("content"), str)
        ):
            raise ValueError("Invalid message or role order in the memory file.")
        history.append({"role": expected_role, "content": message["content"]})

    return summary, history


def save_memory(summary, history):
    """Write a temporary file, then replace the saved conversation atomically."""
    data = {
        "version": 1,
        "model": MODEL,
        "summary": summary,
        "messages": history,
    }
    temporary_file = MEMORY_FILE.with_suffix(".tmp")
    temporary_file.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary_file.replace(MEMORY_FILE)


def main():
    try:
        summary, history = load_memory()
    except (OSError, ValueError) as error:
        # Do not overwrite unreadable or malformed saved memory with an empty chat.
        print(f"Could not load memory: {error}")
        print(f"Check or move this file before trying again: {MEMORY_FILE}")
        return

    print("Local Chatbot - Persistent Summary + Recent History")
    print(f"Memory file: {MEMORY_FILE}")
    print(f"Restored {len(history) // 2} exchanges; summary available: {bool(summary)}")
    print("Uses estimated tokens. Successful turns are saved automatically.")
    print("Commands: /memory, /save, /reset, exit")

    while True:
        try:
            prompt = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            prompt = "exit"

        if prompt.lower() in {"exit", "quit", "/save"}:
            try:
                save_memory(summary, history)
                print("Memory saved.")
            except OSError as error:
                print(f"Memory could not be saved: {error}")
                print("The previous saved file is unchanged; the latest changes are not saved.")
            if prompt.lower() != "/save":
                break
            continue

        if not prompt:
            continue
        if prompt.lower() == "/reset":
            try:
                save_memory("", [])
            except OSError as error:
                print(f"Reset was not saved; existing memory was kept: {error}")
                continue
            summary, history = "", []
            print("Conversation and saved summary cleared. System instruction kept.")
            continue
        if prompt.lower() == "/memory":
            show_memory(summary, history)
            continue

        try:
            new_summary, recent, request = prepare_context(summary, history, prompt)
            print(f"[Estimated input: {estimate_tokens(request)} / {INPUT_BUDGET} tokens]")
            response = ollama.chat(
                model=MODEL,
                messages=request,
                options={"num_ctx": CONTEXT_TOKENS, "num_predict": ANSWER_TOKENS},
            )
            answer = (response.message.content or "").strip()
            if not answer:
                raise ValueError("Ollama returned an empty answer.")
        except Exception as error:
            print(f"\nRequest failed: {error}")
            continue

        summary = new_summary
        history = recent + [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]
        print(f"\nAssistant: {answer}")

        try:
            save_memory(summary, history)
        except OSError as error:
            print(f"Answer received, but memory could not be saved: {error}")
            print("It remains in this process. Fix storage, then use /save to retry.")


if __name__ == "__main__":
    main()
