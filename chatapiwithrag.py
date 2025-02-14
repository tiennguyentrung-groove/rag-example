from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import os
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Initialize OpenAI client
client = OpenAI()

# Load documents for RAG
loader = TextLoader("data/pf-gettingstarted.txt")  # Replace with your document source
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(loader.load(), embedding=embeddings)

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Retrieve relevant context
        context_docs = vectorstore.similarity_search(request.message, k=3)
        context_text = "\n".join([doc.page_content for doc in context_docs])

        # Generate response with RAG
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Use the following context to answer the user query: " + context_text},
                {"role": "user", "content": request.message}
            ]
        )
        return {"response": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run with: uvicorn chatapiwithrag:app --reload