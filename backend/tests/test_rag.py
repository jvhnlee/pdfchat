import pytest
from unittest.mock import patch, MagicMock
import rag_engine

def test_rag_functionality(setup_mock_pdf, mock_embeddings):
    """Verifies document parsing, chunking, and retriever creation."""
    retriever = rag_engine.process_pdf(str(setup_mock_pdf), mock_embeddings)
    assert retriever is not None
    
    # Retrieve documents from the in-memory Chroma instance
    retrieved_docs = retriever.invoke("mock query")
    assert len(retrieved_docs) > 0
    assert any("Mock" in doc.page_content for doc in retrieved_docs)

def test_llm_functionality(mock_llm):
    """Verifies that the LLM chain formats prompt correctly and executes."""
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [
        MagicMock(page_content="Mock content context for prompt.")
    ]
    
    chain = rag_engine.build_qa_chain(mock_llm, mock_retriever)
    response = chain.invoke({"question": "What is the topic?"})
    assert "Mock response" in response


def test_clean_text():
    """Verifies that clean_text removes line-wraps, multiple periods, spaces, and normalizes unicodes."""
    # Multiple periods (double period artifact)
    assert rag_engine.clean_text("7.. **Consistent Wake-Up Time**") == "7. **Consistent Wake-Up Time**"
    
    # Hyphenated line wrap
    assert rag_engine.clean_text("waking-\nup at night") == "wakingup at night"
    
    # Smart quotes normalization (NFKC)
    assert rag_engine.clean_text("“hello” and ‘world’") == '"hello" and \'world\''
    
    # Consolidation of duplicate whitespaces and blank lines removal
    input_text = "\n\nHello   world!  \n\n\nThis is a   test.\n"
    expected = "Hello world!\nThis is a test."
    assert rag_engine.clean_text(input_text) == expected

