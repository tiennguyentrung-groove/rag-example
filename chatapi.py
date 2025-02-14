from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import os

client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1") #ollama
client = OpenAI() #OpenAI

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        response = client.chat.completions.create(
            #model="deepseek-r1:7b",
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": request.message}]
        )
        return {"response": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run with: uvicorn chatapi:app --reload