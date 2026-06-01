import logging
import re
import unicodedata
from typing import Any, List, Optional, Iterator, AsyncIterator
import httpx
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk, ChatResult, ChatGeneration
from langchain_core.callbacks import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
import config

logger = logging.getLogger("pdf_chat.rag_engine")

def clean_text(text: str) -> str:
    """Cleans up common PDF extraction noise and formatting artifacts."""
    if not text:
        return ""
    # Normalize unicode characters (smart quotes, ligatures, etc.)
    text = unicodedata.normalize("NFKC", text)
    
    # Normalize smart quotes to standard straight quotes
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    
    # Remove soft hyphens and combine words split across lines
    text = text.replace("\xad", "").replace("-\n", "")
    
    # Normalize multiple periods (like "7..") to a single period
    text = re.sub(r'\.{2,}', '.', text)
    
    # Normalize excessive whitespaces and duplicate newlines
    lines = [line.strip() for line in text.splitlines()]
    cleaned_lines = [line for line in lines if line]
    text = "\n".join(cleaned_lines)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    
    return text.strip()

class HuggingFaceChatLLM(BaseChatModel):
    repo_id: str
    token: str
    temperature: float = 0.35
    max_new_tokens: int = 1024

    @property
    def _llm_type(self) -> str:
        return "huggingface_chat_llm"

    def _convert_messages(self, messages: List[BaseMessage]) -> List[dict]:
        payload_messages = []
        for msg in messages:
            if msg.type == "system":
                role = "system"
            elif msg.type == "human":
                role = "user"
            elif msg.type == "ai":
                role = "assistant"
            else:
                role = "user"
            payload_messages.append({"role": role, "content": msg.content})
        return payload_messages

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.repo_id,
            "messages": self._convert_messages(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_new_tokens,
            "frequency_penalty": 0.5,
            "repetition_penalty": 1.2
        }
        res = httpx.post(
            "https://router.huggingface.co/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30.0
        )
        res.raise_for_status()
        content = res.json()["choices"][0]["message"]["content"]
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.repo_id,
            "messages": self._convert_messages(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_new_tokens,
            "frequency_penalty": 0.5,
            "repetition_penalty": 1.2,
            "stream": True
        }
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                "https://router.huggingface.co/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30.0
            ) as response:
                response.raise_for_status()
                import json
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            chunk_text = data["choices"][0]["delta"].get("content", "")
                            if chunk_text:
                                yield ChatGenerationChunk(message=AIMessageChunk(content=chunk_text))
                                if run_manager:
                                    await run_manager.on_llm_new_token(chunk_text)
                        except Exception:
                            pass

def get_embeddings() -> HuggingFaceEndpointEmbeddings:
    """Returns the Hugging Face embedding model."""
    token = config.settings.huggingfacehub_api_token
    if not token:
        raise ValueError("Hugging Face API Token (HUGGINGFACEHUB_API_TOKEN) is not set.")
    return HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=token
    )

def get_llm() -> BaseChatModel:
    """Returns the Hugging Face LLM endpoint."""
    token = config.settings.huggingfacehub_api_token
    if not token:
        raise ValueError("Hugging Face API Token (HF_TOKEN / HUGGINGFACEHUB_API_TOKEN) is not set.")
    return HuggingFaceChatLLM(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        token=token,
        temperature=0.35,
        max_new_tokens=1024
    )


def process_pdf(pdf_path: str, embeddings: Any, collection_name: str = "langchain") -> Any:
    """Splits a PDF and returns a Chroma vector store retriever (in-memory).
    
    TODO (Future Improvements):
    - Hybrid Search & Re-ranking: Combine semantic search (Chroma) with keyword-based retrieval (BM25)
      and apply a re-ranker model (e.g., Cohere/Cross-Encoder) to order context passages.
    - Persistent Vector Database: Shift to a persistent directory-based store (or external provider
      like Pinecone/pgvector) so session databases are retained.
    - Advanced Semantic Chunking: Upgrade RecursiveCharacterTextSplitter to layout-aware chunking
      or semantic-aware chunking based on embedding similarities.
    """
    logger.info(f"Loading PDF from {pdf_path}")
    
    reader = PdfReader(pdf_path)
    docs = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            cleaned = clean_text(text)
            if cleaned:
                docs.append(Document(page_content=cleaned, metadata={"source": pdf_path, "page": i}))
            
    if not docs:
        raise ValueError("The uploaded PDF is empty or could not be read.")

    # Medium chunk size, small overlap
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
    chunks = splitter.split_documents(docs)
    logger.info(f"Created {len(chunks)} chunks from PDF")

    # In-memory Chroma database instance from langchain_chroma
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name
    )
    
    # Similarity search retriever returning top 3 relevant chunks
    return vectorstore.as_retriever(search_kwargs={"k": 3})

def format_docs(docs: List[Any]) -> str:
    """Combines document page contents into a clean text block."""
    return "\n\n".join(doc.page_content for doc in docs)

def build_qa_chain(llm: Any, retriever: Any) -> Any:
    """Assembles the LCEL QA chain with a custom, emoji-free prompt template.
    
    TODO (Future Improvement: Response Refinement): 
    - Implement a reflection/refinement step. For example, pipe the output of this chain into a 
      validator LLM/prompter that checks the answer against retrieved documents to ensure no 
      hallucinations, fixes markdown formatting issues, and cleans up references.
    - Fix repetitions and grammatical errors.
    """
    qa_system_prompt = (
        "You are a helpful research assistant. "
        "Use the following retrieved context chunks to answer the user's question. "
        "If you do not know the answer, say that you do not know. "
        "Do not make up information.\n\n"
        "Context:\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])
    
    chain = (
        {
            "context": (lambda x: x["question"]) | retriever | format_docs,
            "question": lambda x: x["question"],
            "chat_history": lambda x: x.get("chat_history", [])
        }
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    return chain

