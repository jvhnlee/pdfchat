import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from main import app, VECTOR_STORES

client = TestClient(app)

@pytest.fixture(autouse=True)
def mock_rag_components(request, mock_embeddings, mock_llm):
    """Automatically patch get_embeddings and get_llm for all system tests except real API tests."""
    if "real_api" in request.keywords:
        yield
    else:
        with patch("rag_engine.get_embeddings", return_value=mock_embeddings), \
             patch("rag_engine.get_llm", return_value=mock_llm):
            yield

def test_successful_rag_pipeline(setup_mock_pdf):
    """Test Case 1: Upload a PDF and query the chat API successfully (normal flow)."""
    session_id = "test-session-1"
    
    # Ingest PDF
    with open(setup_mock_pdf, "rb") as f:
        files = {"file": ("mock_document.pdf", f, "application/pdf")}
        data = {"session_id": session_id}
        upload_resp = client.post("/api/upload", files=files, data=data)
    
    assert upload_resp.status_code == 200
    assert upload_resp.json()["status"] == "success"
    assert session_id in VECTOR_STORES

    # Chat with RAG context
    chat_payload = {"session_id": session_id, "question": "What does the PDF say?"}
    chat_resp = client.post("/api/chat", json=chat_payload)
    
    assert chat_resp.status_code == 200
    assert len(chat_resp.text) > 0
    assert "Mock response" in chat_resp.text

def test_missing_document_chat():
    """Test Case 2: Querying chat prior to uploading a document returns a 404 error."""
    session_id = "non-existent-session"
    chat_payload = {"session_id": session_id, "question": "Hello?"}
    chat_resp = client.post("/api/chat", json=chat_payload)
    
    assert chat_resp.status_code == 404
    assert "No document has been uploaded" in chat_resp.json()["detail"]

def test_invalid_file_type_upload(tmp_path):
    """Test Case 3: Uploading a non-PDF file returns a 400 error."""
    session_id = "test-session-invalid-file"
    text_file = tmp_path / "mock.txt"
    text_file.write_text("Not a PDF file")

    with open(text_file, "rb") as f:
        files = {"file": ("mock.txt", f, "text/plain")}
        data = {"session_id": session_id}
        upload_resp = client.post("/api/upload", files=files, data=data)
    
    assert upload_resp.status_code == 400
    assert "Only PDF files are supported" in upload_resp.json()["detail"]

def test_missing_session_id_upload(setup_mock_pdf):
    """Test Case 4: Uploading without a session ID returns a 422/400 error."""
    with open(setup_mock_pdf, "rb") as f:
        files = {"file": ("mock_document.pdf", f, "application/pdf")}
        # Missing data parameter
        upload_resp = client.post("/api/upload", files=files)
    
    # FastAPI returns 422 Unprocessable Entity when required form fields are missing
    assert upload_resp.status_code == 422

def test_missing_session_id_chat():
    """Test Case 5: Querying chat without a session ID returns a 422 error."""
    chat_payload = {"question": "Hello?"} # Missing session_id
    chat_resp = client.post("/api/chat", json=chat_payload)
    
    assert chat_resp.status_code == 422

def test_clear_session_history(setup_mock_pdf):
    """Test Case 6: Clearing session history works successfully."""
    session_id = "test-session-clear"
    
    # Ingest PDF
    with open(setup_mock_pdf, "rb") as f:
        files = {"file": ("mock_document.pdf", f, "application/pdf")}
        data = {"session_id": session_id}
        client.post("/api/upload", files=files, data=data)
        
    # Simulate chat
    VECTOR_STORES[session_id]["history"] = [("user", "Hello"), ("assistant", "Hi")]
    
    # Clear history
    clear_resp = client.post("/api/clear", json={"session_id": session_id, "question": ""})
    assert clear_resp.status_code == 200
    assert len(VECTOR_STORES[session_id]["history"]) == 0


@pytest.mark.real_api
def test_real_api_rag_pipeline(setup_mock_pdf):
    """Test Case 7: Upload a PDF and query the chat API using the real Hugging Face API and Qwen model."""
    import os
    import dotenv
    import uuid
    import config
    
    # Load the real environment variables from backend/.env
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    dotenv.load_dotenv(dotenv_path=env_path, override=True)
    
    real_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not real_token or real_token.startswith("mock"):
        pytest.skip("Real Hugging Face API token is not available in environment/env file.")

    original_token = config.settings.huggingfacehub_api_token
    config.settings.huggingfacehub_api_token = real_token
    os.environ["HF_TOKEN"] = real_token
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = real_token
    
    try:
        session_id = f"real-test-session-{uuid.uuid4()}"
        
        # Ingest PDF
        with open(setup_mock_pdf, "rb") as f:
            files = {"file": ("mock_document.pdf", f, "application/pdf")}
            data = {"session_id": session_id}
            upload_resp = client.post("/api/upload", files=files, data=data)
        
        assert upload_resp.status_code == 200
        assert upload_resp.json()["status"] == "success"
        
        # Chat with real LLM
        chat_payload = {"session_id": session_id, "question": "What is the topic of the document?"}
        chat_resp = client.post("/api/chat", json=chat_payload)
        
        assert chat_resp.status_code == 200
        response_text = chat_resp.text
        assert len(response_text) > 0
        assert "[Error generating response" not in response_text
        print(f"\nReal API Streamed Response: {response_text.strip()}")
    finally:
        config.settings.huggingfacehub_api_token = original_token





