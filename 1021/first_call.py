import ollama

MODEL = "gemma3:1b"


def main():
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": "Say hello in one sentence."}],
    )
    print(response.message.content)


if __name__ == "__main__":
    main()
