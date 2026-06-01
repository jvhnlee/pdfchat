import os
import sys
import pytest
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from unittest.mock import MagicMock
from langchain_core.runnables import Runnable
from langchain_core.messages import AIMessage, AIMessageChunk

# Add workspace root to python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Override config environment variables before imports
os.environ["PORT"] = "8000"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "mock-hf-token"

@pytest.fixture(scope="session", autouse=True)
def setup_mock_pdf():
    """Generates a simple mock PDF document for tests."""
    pdf_path = Path(__file__).parent / "mock_document.pdf"
    if not pdf_path.exists():
        c = canvas.Canvas(str(pdf_path), pagesize=letter)
        c.drawString(100, 750, "Mock PDF context for testing.")
        c.drawString(100, 720, "This contains RAG details and data.")
        c.showPage()
        c.save()
    return pdf_path

@pytest.fixture
def mock_embeddings():
    """Returns a mock embeddings object."""
    mock = MagicMock()
    mock.embed_documents.side_effect = lambda texts: [[0.1] * 128 for _ in texts]
    mock.embed_query.return_value = [0.1] * 128
    mock.__class__.__name__ = "MockEmbeddings"
    return mock

class MockLLM(Runnable):
    """Custom LLM runnable subclass to mock streaming/invocation correctly."""
    def invoke(self, input, config=None, **kwargs):
        return "Mock response based on context."

    async def astream(self, input, config=None, **kwargs):
        for chunk in ["Mock ", "response ", "stream."]:
            yield chunk

    def stream(self, input, config=None, **kwargs):
        for chunk in ["Mock ", "response ", "stream."]:
            yield chunk

@pytest.fixture
def mock_llm():
    """Returns a mock LLM runnable."""
    return MockLLM()
