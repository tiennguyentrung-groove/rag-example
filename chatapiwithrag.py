from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import uvicorn

# Initialize client
#client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1") #ollama
client = OpenAI()

#embeddings = OllamaEmbeddings(model = "llama3.2")
embeddings = OpenAIEmbeddings()


app = FastAPI()
vectorstore = FAISS.load_local("pf_index", embeddings, allow_dangerous_deserialization=True);

class ChatRequest(BaseModel):
    message: str

@app.get('/index')
async def index():
    try:
        # Load documents from all file in folder data for RAG 
        #loader = TextLoader("data/pf-gettingstarted.txt") 
        loader = DirectoryLoader("data", glob="*.txt")
        documents = loader.load()

        
        # We can use FAISS (Facebook AI Similarity Search) or ChromaDB or InMemoryVectorStore for vector store
        new_vectorstore = FAISS.from_documents(documents, embedding=embeddings)

        # Save the FAISS index for later use
        new_vectorstore.save_local("pf_index")

        return {"response": "OK!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Retrieve relevant context
        context_docs = vectorstore.similarity_search(request.message, k=3)
        context_text = "\n".join([doc.page_content for doc in context_docs])

        # Generate response with RAG
        response = client.chat.completions.create(
            model="gpt-4o-mini", #
            messages=[
                {"role": "system", "content": "Use the following context to answer the user query: " + context_text},
                {"role": "user", "content": request.message}
            ]
        )
        return {"response": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

uvicorn.run(app, host="0.0.0.0", port=8000)

# Run with: python .\chatapiwithrag.py