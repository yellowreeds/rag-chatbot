"""November 4, version 2: a token-budgeted summary plus recent history.

Run: python chatbot_02_summary_tokens.py
Requires the existing rag-class environment, ollama library, and gemma3:1b.
Memory is temporary: exiting this script clears it.

For teaching, estimate 1 token per 4 characters, plus message overhead.
This is NOT the model's tokenizer; estimates vary with language and content.
The safety margin helps, but does not guarantee an exact model-token limit.
"""

import ollama

from chatbot_memory import (
    ANSWER_TOKENS,
    CONTEXT_TOKENS,
    INPUT_BUDGET,
    MODEL,
    estimate_tokens,
    prepare_context,
    show_memory,
)


def main():
    summary = ""
    history = []  # Complete user/assistant pairs; instructions stay separate.

    print("Local Chatbot - Summary + Recent History")
    print("Uses estimated tokens; older exchanges are summarized when needed.")
    print("Commands: /memory, /reset, exit")

    while True:
        try:
            prompt = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if prompt.lower() in {"exit", "quit"}:
            break
        if not prompt:
            continue
        if prompt.lower() == "/reset":
            summary, history = "", []
            print("Conversation and summary cleared. System instruction kept.")
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

        # Commit memory only after both compression and answering succeed.
        summary = new_summary
        history = recent + [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]
        print(f"\nAssistant: {answer}")


if __name__ == "__main__":
    main()
