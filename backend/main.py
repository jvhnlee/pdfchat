import os
import uuid
import tempfile
import logging
from typing import Any
import httpx2
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage

import config
import rag_engine

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pdf_chat.main")

# In-memory dictionary to store session-specific retrievers and histories
# TODO (Future Improvements):
# - Persistent session databases (instead of in-memory dictionaries that reset on restart).
# - Multi-document collections: Allow a single session ID to ingest and query across multiple vector stores or files.
# - Authentication & security: Secure access with user JWT tokens to prevent unauthorized session ID lookups.
VECTOR_STORES = {}

# FastAPI App
app = FastAPI(
    title="pdfchat",
    description="Minimal stateless PDF RAG chatbot."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    session_id: str
    question: str

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

@app.post("/api/clear")
async def api_clear(req: ChatRequest):
    """Clears conversation history for a session."""
    session_id = req.session_id.strip()
    if session_id in VECTOR_STORES:
        VECTOR_STORES[session_id]["history"] = []
        logger.info(f"Cleared history for session {session_id}")
    return {"status": "success"}

@app.post("/api/upload")
async def api_upload(
    file: UploadFile = File(...),
    session_id: str = Form(...)
):
    """Processes the uploaded PDF, chunks and indexes it in-memory, and registers the session."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported."
        )
    if not session_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session ID cannot be empty."
        )

    # Save to a temporary file locally to parse it
    fd, temp_path = tempfile.mkstemp(suffix=".pdf")
    try:
        content = await file.read()
        with os.fdopen(fd, "wb") as tmp:
            tmp.write(content)
        
        # Instantiate embeddings
        try:
            embeddings = rag_engine.get_embeddings()
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Configuration error: {str(e)}"
            )

        # Process the PDF using chunking size=800, overlap=80
        try:
            retriever = rag_engine.process_pdf(temp_path, embeddings, collection_name=session_id)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to process PDF document: {str(e)}"
            )

        # Save retriever and blank history list for the session
        VECTOR_STORES[session_id] = {
            "retriever": retriever,
            "history": []
        }
        logger.info(f"Processed PDF '{file.filename}' for session {session_id}")
        return {"status": "success", "filename": file.filename}

    finally:
        # Secure cleanup of temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/api/chat")
async def api_chat(req: ChatRequest):
    """Retrieves document context and streams response chunks from Hugging Face LLM."""
    session_id = req.session_id.strip()
    question = req.question.strip()

    if not session_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session ID is required."
        )
    if not question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question is required."
        )

    session_data = VECTOR_STORES.get(session_id)
    if not session_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No document has been uploaded for this session. Please upload a PDF first."
        )

    retriever = session_data["retriever"]
    history = session_data["history"]

    # Convert history tuples to LangChain message formats
    lc_history = []
    for role, text in history:
        if role == "user":
            lc_history.append(HumanMessage(content=text))
        elif role == "assistant":
            lc_history.append(AIMessage(content=text))

    try:
        llm = rag_engine.get_llm()
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Configuration error: {str(e)}"
        )

    try:
        chain = rag_engine.build_qa_chain(llm, retriever)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chain build failed: {str(e)}"
        )

    async def stream_response():
        full_response = ""
        try:
            async for chunk in chain.astream({"question": question, "chat_history": lc_history}):
                full_response += chunk
                yield chunk
            # Save transaction to history list
            history.append(("user", question))
            history.append(("assistant", full_response))
        except Exception as e:
            logger.error(f"Error during streaming response: {str(e)}")
            yield f"\n\n[Error generating response: {str(e)}]"

    return StreamingResponse(stream_response(), media_type="text/plain")


if __name__ == "__main__":
    logger.info(f"Starting server on port {config.settings.port}...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=config.settings.port,
        reload=False
    )
