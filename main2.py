import os
import asyncio
import requests
import uvicorn
from fastapi import FastAPI, Depends, HTTPException, status, Header
from pydantic import BaseModel, Field, HttpUrl
from typing import List
from urllib.parse import urlparse
from dotenv import load_dotenv

# LangChain Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_mistralai import ChatMistralAI
from langsmith import traceable

# Load environment variables
load_dotenv()

# --- Configuration ---
EXPECTED_API_KEY = os.getenv("API_KEY", "your_default_secret_api_key")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
GEMINI_API_KEYS = os.getenv("GOOGLE_API_KEYS", "").split(",")

# --- Gemini Key Manager ---
class GeminiKeyManager:
    def __init__(self, api_keys: List[str]):
        self.api_keys = api_keys
        self.index = 0

    def get_key(self):
        return self.api_keys[self.index]

    def rotate_key(self):
        self.index = (self.index + 1) % len(self.api_keys)
        return self.get_key()

key_manager = GeminiKeyManager(GEMINI_API_KEYS)

def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=key_manager.get_key(),
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=0,  # must be present
    )


# --- Embeddings ---
model = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEndpointEmbeddings(
    model=model,
    task="feature-extraction",
    huggingfacehub_api_token=HF_TOKEN,
)

# --- FastAPI App Initialization ---
app = FastAPI(title="Optimized Insurance Policy Q&A API", version="1.1.0")

# --- Pydantic Models ---
class APIRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class APIResponse(BaseModel):
    answers: List[str]

# --- Authentication ---
async def verify_api_key(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication scheme.")
    token = authorization.split(" ")[1]
    if token != EXPECTED_API_KEY:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API Key.")

# --- File Helpers ---
def get_safe_filename_from_url(url):
    path = urlparse(url).path
    return f"temp_{os.path.basename(path)}"

# --- Document Processing ---
@traceable
def process_document_and_get_retriever(pdf_url: str):
    response = requests.get(pdf_url)
    response.raise_for_status()
    temp_pdf_path = get_safe_filename_from_url(pdf_url)
    with open(temp_pdf_path, "wb") as f:
        f.write(response.content)

    loader = PyPDFLoader(temp_pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    all_splits = splitter.split_documents(docs)

    vector_store = FAISS.from_documents(all_splits, embedding=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    os.remove(temp_pdf_path)
    return retriever

# --- Prompt and Chain ---
prompt = ChatPromptTemplate.from_template("""
You are an expert insurance claims processor. Answer the question in Text only using the context provided in the Documents. If you can't infer from the Documents, say that the information is not available in the provided context.

Text: {query}
Documents:
{context}
""")

# --- Async Chain Execution with Key Rotation ---
@traceable
async def safe_llm_call(query: str, context_docs: List[str]) -> str:
    for _ in range(len(GEMINI_API_KEYS)):
        try:
            chain = prompt | get_llm()
            result = await chain.ainvoke({"query": query, "context": context_docs})
            # If successful, return result content
            return getattr(result, 'content', result)
        except Exception as e:
            print("\n\n\n\n\n\n\n\n\n")
            # Check if it is a quota/rate exception
            if "rate limit" in str(e).lower() or "quota" in str(e).lower() \
               or "429" in str(e):  # Google often returns 429 for quota exceeded
                key_manager.rotate_key()
                continue  # Try next key
            raise  # Other error: propagate
    raise RuntimeError("All Gemini API keys exhausted.")


@traceable
async def answer_question(query: str, retriever):
    context_docs = await asyncio.to_thread(retriever.invoke, query)
    return await safe_llm_call(query, context_docs)

# --- Endpoint ---
@app.post("/api/v1/hackrx/run", response_model=APIResponse, dependencies=[Depends(verify_api_key)])
@traceable
async def hackrx_run(request: APIRequest):
    try:
        retriever = process_document_and_get_retriever(str(request.documents))
        tasks = [answer_question(q, retriever) for q in request.questions]
        answers = await asyncio.gather(*tasks)
        return APIResponse(answers=answers)
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download document: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
