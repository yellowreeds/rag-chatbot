"""November 4, version 1: remember only recent complete exchanges.

Starting point: ../1021/chatbot.py
Run: python chatbot_01_sliding_window.py
Requires the existing rag-class environment, ollama library, and gemma3:1b.
Memory is temporary: exiting this script clears it.
"""

import ollama

MODEL = "gemma3:1b"
SYSTEM_PROMPT = "Be helpful and concise."
MAX_EXCHANGES = 5  # One exchange = one user message + one assistant reply.


def main():
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("Local Chatbot - Sliding Window")
    print(f"Keep the last {MAX_EXCHANGES} complete exchanges.")
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
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("Conversation cleared. System instruction kept.")
            continue

        if prompt.lower() == "/memory":
            print(f"Stored exchanges: {(len(messages) - 1) // 2}")
            for message in messages[1:]:
                print(f"{message['role']}: {message['content']}")
            continue

        messages.append({"role": "user", "content": prompt})

        try:
            response = ollama.chat(model=MODEL, messages=messages)
            answer = (response.message.content or "").strip()
            if not answer:
                raise ValueError("Ollama returned an empty answer.")
        except Exception as error:
            # Remove the unanswered user turn, as in the October 21 code.
            messages.pop()
            print(f"\nRequest failed: {error}")
            continue

        print(f"\nAssistant: {answer}")
        messages.append({"role": "assistant", "content": answer})

        # Keep the system instruction and the latest complete exchanges.
        # The next request adds the current question to these retained turns.
        recent_history = messages[1:][-MAX_EXCHANGES * 2:]
        messages = [messages[0]] + recent_history


if __name__ == "__main__":
    main()
