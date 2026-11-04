import ollama

response = ollama.chat(
   model="gemma3:1b",
   messages=[
       {"role": "user",
        "content": "Say hello in one sentence."}
   ],
)

print(response.message.content)