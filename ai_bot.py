#AI chat using OPENAI or Ollama

import os
from openai import OpenAI

#client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1") #ollama
client = OpenAI() #OpenAI - API key is stored in environment variable "OPEN_API_KEY"

def chat_with_ai(prompt):
    response = client.chat.completions.create(
        #model="deepseek-r1:7b",
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0.7,
    )
    print(response.choices[0].message.content)
    return response.choices[0].message.content


chat_with_ai("Website stackoverflow được phát triển từ lúc nào?")
