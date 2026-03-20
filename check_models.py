import os
from openai import OpenAI

api_key = os.environ.get("NVIDIA_API_KEY", "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")

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key
)

# Скрипт сам создаст файл и запишет в него всё в UTF-8
with open("models.txt", "w", encoding="utf-8") as f:
    f.write("Все доступные модели в NVIDIA NIM:\n\n")
    
    for m in client.models.list().data:
        # Теперь мы выводим абсолютно все модели (LLM, Embeddings, Vision и т.д.)
        f.write(f" - {m.id}\n")

print("Done! Open models.txt to see the full list.")