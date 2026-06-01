import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_upload_non_pdf_edge_case():
    """Edge Case 1: Uploading a non-PDF file should return 400 Bad Request."""
    files = {"file": ("test.txt", b"some text content", "text/plain")}
    data = {"session_id": "session-123"}
    response = client.post("/api/upload", files=files, data=data)
    assert response.status_code == 400
    assert "Only PDF files are supported" in response.json()["detail"]

def test_chat_empty_session_id_edge_case():
    """Edge Case 2: Querying chat with an empty session ID should return 400 Bad Request."""
    payload = {"session_id": "", "question": "What is the summary?"}
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 400
    assert "Session ID is required" in response.json()["detail"]
