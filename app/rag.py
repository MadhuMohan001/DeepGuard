"""RAG over your research paper / report / forensics notes using LangChain + ChromaDB."""
import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCS_DIR = Path(os.getenv("DOCS_DIR", "docs"))
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-5-5")

SYSTEM_PROMPT = (
    "You explain results from a multimodal deepfake detector. Answer ONLY using the "
    "context provided. If the context does not contain the answer, say you don't know. "
    "Keep answers short and in plain language."
)


def _read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8", errors="ignore")


class RagEngine:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        self.store = Chroma(
            collection_name="deepguard",
            embedding_function=self.embeddings,
            persist_directory=CHROMA_DIR,
        )

    def ingest(self) -> int:
        """Chunk every .pdf/.txt/.md in docs/ and add it to ChromaDB."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
        texts, metas = [], []
        for path in sorted(DOCS_DIR.glob("*")):
            if path.suffix.lower() not in {".pdf", ".txt", ".md"}:
                continue
            for i, chunk in enumerate(splitter.split_text(_read_file(path))):
                texts.append(chunk)
                metas.append({"source": path.name, "chunk": i})
        if texts:
            self.store.add_texts(texts=texts, metadatas=metas)
        return len(texts)

    def retrieve(self, query: str, k: int = 4):
        return self.store.similarity_search(query, k=k)

    def answer(self, question: str, k: int = 4) -> dict:
        docs = self.retrieve(question, k)
        context = "\n\n".join(d.page_content for d in docs)
        sources = sorted({d.metadata.get("source", "?") for d in docs})
        return {"answer": self._llm(question, context), "sources": sources}

    def explain_prediction(self, prediction: dict) -> dict:
        question = (
            f"The detector classified a video as {prediction['label']} with "
            f"{prediction['confidence']:.0%} confidence. Explain in plain language what "
            "this means and what cues or limitations of the method are relevant."
        )
        return self.answer(question)

    def _llm(self, question: str, context: str) -> str:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            # Works without a key so you can test retrieval; set the key for real answers.
            return "[LLM key not set] Retrieved context:\n" + context[:1200]
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model=LLM_MODEL,
            max_tokens=500,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}],
        )
        return "".join(b.text for b in msg.content if b.type == "text")
