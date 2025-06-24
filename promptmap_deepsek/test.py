import openai

openai.api_base = "https://api.deepseek.com"
openai.api_key = "sk-a7bbb31a87324a1394621e322f57d12f"

response = openai.ChatCompletion.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
)

print(response.choices[0].message["content"])
