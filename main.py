import os
import asyncio
import requests
import uvicorn
from fastapi import FastAPI, Depends, HTTPException, status, Header
from pydantic import BaseModel, Field, HttpUrl
from typing import List
from urllib.parse import urlparse
import os


# Import your existing RAG components
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langsmith import traceable
from dotenv import load_dotenv
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env file
load_dotenv()
# --- Configuration ---
EXPECTED_API_KEY = os.getenv("API_KEY", "your_default_secret_api_key")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Optimized Insurance Policy Q&A API",
    description="An API that answers questions about an insurance policy document using a fast, cached RAG pipeline.",
    version="1.1.0"
)

# --- Pydantic Models ---
class APIRequest(BaseModel):
    documents: HttpUrl = Field(..., description="URL to the policy PDF document.")
    questions: List[str] = Field(..., description="A list of questions to answer based on the document.")

class APIResponse(BaseModel):
    answers: List[str] = Field(..., description="A list of answers corresponding to the questions.")

# --- Authentication ---
async def verify_api_key(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication scheme.")
    token = authorization.split(" ")[1]
    if token != EXPECTED_API_KEY:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API Key.")
    

def get_safe_filename_from_url(url):
    path = urlparse(url).path
    filename = os.path.basename(path)
    return f"temp_{filename}"

# --- OPTIMIZATION 1: Caching Mechanism ---
# This dictionary will store our retrievers in memory to avoid reprocessing.

@traceable
def process_document_and_get_retriever(pdf_url: str):
    """
    Loads, splits a PDF, and creates a retriever.
    Crucially, it caches the retriever based on the PDF URL.
    """
    print(f"CACHE MISS: Processing new document from {pdf_url}.")
    
    # Download the PDF
    response = requests.get(pdf_url)
    response.raise_for_status()
    temp_pdf_path = get_safe_filename_from_url(pdf_url)
    with open(temp_pdf_path, "wb") as f:
        f.write(response.content)

    # 1. Load the document
    loader = PyPDFLoader(temp_pdf_path)
    docs = loader.load()

    # 2. Split the document
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    all_splits = text_splitter.split_documents(docs)

    # 3. Create and store embeddings (This is the slow part we want to cache)
    vector_store = Chroma.from_documents(
        documents=all_splits,
        embedding = MistralAIEmbeddings(
    model="mistral-embed",  # This is the standard embedding model name
    mistral_api_key=MISTRAL_API_KEY  # Optional if set in environment variable
)
    )

    # 4. Create a retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # 5. Store the new retriever in the cache and clean up
    os.remove(temp_pdf_path)
    
    return retriever

# --- LLM Chain Definition ---
structured_input_template = """
You are an expert insurance claims processor. Answer the question in Text only using the context provided in the Documents. If you can't infer from the Documents, say that the information is not available in the provided context.

Text: {query}
Documents:
{context}
"""
prompt = ChatPromptTemplate.from_template(structured_input_template)
# model = ChatGoogleGenerativeAI(
#     model="gemini-1.5-pro",
#     google_api_key=GEMINI_API_KEY,
#     temperature=0.2
# )
model = ChatMistralAI(model="mistral-medium", mistral_api_key=MISTRAL_API_KEY)
chain = prompt | model

# --- OPTIMIZATION 2: Asynchronous Answering ---
async def answer_question(query: str, retriever, chain: Runnable):
    """Answers a single question asynchronously."""
    # The retriever's 'invoke' might be blocking, run it in a thread pool
    context_docs = await asyncio.to_thread(retriever.invoke, query)
    
    # The chain's 'ainvoke' is the async version for network calls
    result = await chain.ainvoke({"query": query, "context": context_docs})
    return result.content

# --- API Endpoint ---
@app.post(
    "/api/v1/hackrx/run",
    response_model=APIResponse,
    summary="Process a policy document and answer questions",
    dependencies=[Depends(verify_api_key)]
)
async def hackrx_run(request: APIRequest):
    """
    This endpoint receives a URL to a PDF document and a list of questions.
    It processes the document efficiently using caching and answers questions in parallel.
    """
    try:
        # Get the retriever (from cache or by processing the doc)
        # This is a synchronous function, so it's called directly
        retriever = process_document_and_get_retriever(str(request.documents))

        # Create a list of asynchronous tasks for each question
        tasks = [answer_question(query, retriever, chain) for query in request.questions]
        
        # Run all tasks concurrently and wait for them to complete
        answers = await asyncio.gather(*tasks)

        return APIResponse(answers=answers)

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to download document: {e}")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An internal error occurred: {e}")

# --- Main execution block to run the server ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)