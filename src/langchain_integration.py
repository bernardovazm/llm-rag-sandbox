from typing import Tuple, Any, List
from src.rag_pipeline import build_index

try:
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.docstore.document import Document
except Exception:
    HuggingFaceEmbeddings = None
    FAISS = None
    Document = None


def build_langchain_faiss(root_dir: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Tuple[Any, Any]:
    if HuggingFaceEmbeddings is None or FAISS is None:
        raise RuntimeError("Missing langchain or embedding backends; install langchain and transformers")

    docs = build_index(root_dir)
    texts = [d.get("text", "") for d in docs]
    metadatas = [{"path": d.get("path")} for d in docs]

    embed = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = FAISS.from_texts(texts, embed, metadatas=metadatas)
    return vectorstore, embed


def query_langchain(vectorstore: Any, query: str, k: int = 3) -> List[dict]:
    results = vectorstore.similarity_search(query, k=k)
    return results
