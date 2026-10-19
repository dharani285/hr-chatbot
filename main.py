"""
FastAPI app for the HR RAG Chatbot.

Endpoints:
- GET /health: Basic health/status check.
- POST /ask: Answer a question using retrieval + Gemini generation.

Requires environment variable GEMINI_API_KEY to be set (or present in .env).
Run:
    uvicorn main:app --reload --port 8000
"""

import os
import sys
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Allow importing from src/
ROOT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src.retrieve import search  # noqa: E402
from src.cli import format_context, generate_answer  # noqa: E402


class AskRequest(BaseModel):
    question: str = Field(..., description="User question to answer")
    k: int = Field(5, ge=1, le=20, description="Number of chunks to retrieve")
    return_context: bool = Field(True, description="Include concatenated context in response")
    return_sources: bool = Field(True, description="Include per-chunk sources in response")


class SourceItem(BaseModel):
    score: float
    chunk: str
    doc_id: str
    chunk_id: int


class AskResponse(BaseModel):
    answer: str
    context: Optional[str] = None
    sources: Optional[List[SourceItem]] = None


app = FastAPI(title="HR RAG Chatbot API", version="1.0.0")
@app.get("/")
def home():
    return {"message": "HR Chatbot API is running"}

@app.get("/health")
def health():
    """Return server status and artifact presence flags."""
    artifacts_dir = os.path.join(ROOT_DIR, "artifacts")
    return {
        "status": "ok",
        "artifacts_dir": artifacts_dir,
        "has_index": os.path.exists(os.path.join(artifacts_dir, "faiss.index")),
        "has_metadata": os.path.exists(os.path.join(artifacts_dir, "metadata.pkl")),
        "has_embeddings": os.path.exists(os.path.join(artifacts_dir, "embeddings.pkl")),
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """Answer a question using RAG pipeline.

    1) Retrieve top-k relevant chunks
    2) Format context
    3) Generate answer via Gemini model
    """
    try:
        # Make sure env vars from .env are available for generate_answer
        load_dotenv()

        results = search(req.question, k=req.k)
        ctx = format_context(results)
        answer = generate_answer(req.question, ctx)

        payload: AskResponse = AskResponse(answer=answer)
        if req.return_context:
            payload.context = ctx
        if req.return_sources:
            payload.sources = [
                SourceItem(
                    score=score,
                    chunk=chunk,
                    doc_id=md.get("doc_id", ""),
                    chunk_id=int(md.get("chunk_id", -1)),
                )
                for (score, chunk, md) in results
            ]
        return payload
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))