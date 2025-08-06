import os
import requests
import uvicorn
from fastapi import FastAPI, Depends, HTTPException, status, Header
from pydantic import BaseModel, Field, HttpUrl
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable
from dotenv import load_dotenv

load_dotenv()

EXPECTED_API_KEY = os.getenv("API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

app = FastAPI(
    title="Insurance Policy Q&A API",
    description="An API that answers questions about an insurance policy document using a RAG pipeline.",
    version="1.0.0"
)

class APIRequest(BaseModel):
    documents: HttpUrl = Field(..., description="URL to the policy PDF document.")
    questions: List[str] = Field(..., description="A list of questions to answer based on the document.")

class APIResponse(BaseModel):
    answers: List[str] = Field(..., description="A list of answers corresponding to the questions.")

async def verify_api_key(authorization: str = Header(...)):
    """Dependency to verify the bearer token."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication scheme.",
        )
    token = authorization.split(" ")[1]
    if token != EXPECTED_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API Key.",
        )


@traceable
def process_document_and_get_retriever(pdf_path: str):
    """
    Loads, splits a PDF, and creates a retriever.
    """
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)

    vector_store = Chroma.from_documents(
        documents=all_splits,
        embedding=MistralAIEmbeddings(model="mistral-embed", mistral_api_key=MISTRAL_API_KEY)
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    return retriever

structured_input_template = """
You are an expert insurance claims processor.

Answer the question in Text only using the context provided in the Documents.
If you can't infer the answer from the Documents, say that the information is not available in the provided context. Do not make up information.

Text: {query}

Documents:
{context}
"""

prompt = ChatPromptTemplate.from_template(structured_input_template)

model = ChatMistralAI(
    model="mistral-large-latest",
    mistral_api_key=MISTRAL_API_KEY
)

chain = prompt | model

@app.post(
    "/hackrx/run",
    response_model=APIResponse,
    summary="Process a policy document and answer questions",
    dependencies=[Depends(verify_api_key)]
)
async def hackrx_run(request: APIRequest):
    """
    This endpoint receives a URL to a PDF document and a list of questions.
    It processes the document and returns answers using a RAG model.
    """
    try:
        pdf_response = requests.get(str(request.documents))
        pdf_response.raise_for_status()  

        temp_pdf_path = "temp_policy.pdf"
        with open(temp_pdf_path, "wb") as f:
            f.write(pdf_response.content)

        retriever = process_document_and_get_retriever(temp_pdf_path)

        answers = []
        for query in request.questions:
            context_docs = retriever.invoke(query)
            
            result = chain.invoke({"query": query, "context": context_docs})
            
            answers.append(result.content)

        os.remove(temp_pdf_path)

        return APIResponse(answers=answers)

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to download document: {e}")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An internal error occurred: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)