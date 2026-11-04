import ollama

MODEL = "gemma3:1b"
messages = [
    {
        "role": "system",
        "content": "Be helpful and concise."
    }
]

print("Local Chatbot — type exit to quit")

while True:
    prompt = input("\nYou: ").strip()

    if prompt.lower() in {"exit", "quit"}:
        break

    if not prompt:
        continue

    messages.append({
        "role": "user",
        "content": prompt
    })

    try:
        response = ollama.chat(
            model=MODEL,
            messages=messages
        )
        answer = response.message.content

    except Exception as error:
        messages.pop()
        print(f"\nRequest failed: {error}")
        continue

    print(f"\nAssistant: {answer}")

    messages.append({
        "role": "assistant",
        "content": answer
    })