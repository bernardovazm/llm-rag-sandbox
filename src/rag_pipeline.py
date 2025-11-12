from pathlib import Path
from typing import List, Dict, Any, Optional
import re

import numpy as np

try:
    import faiss
except Exception:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def build_index(root_dir: str) -> List[Dict[str, Any]]:
    p = Path(root_dir)
    docs = []
    for f in p.rglob("*.txt"):
        text = _read_text(f)
        docs.append({"path": str(f), "text": text})
    return docs


def _tokenize(s: str) -> List[str]:
    return re.findall(r"\w+", s.lower())


def query_index(query: str, index: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
    qtokens = set(_tokenize(query))
    scored = []
    for doc in index:
        tokens = set(_tokenize(doc.get("text", "")))
        score = len(qtokens & tokens)
        scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for s, d in scored[:top_k]]


def build_faiss_index(root_dir: str, model_name: str = "all-MiniLM-L6-v2") -> tuple:
    if SentenceTransformer is None or faiss is None:
        raise RuntimeError("Missing dependencies: install sentence-transformers and faiss-cpu")

    docs = build_index(root_dir)
    texts = [d.get("text", "") for d in docs]
    embed_model = SentenceTransformer(model_name)
    embeddings = embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    if embeddings.ndim == 1:
        embeddings = np.expand_dims(embeddings, 0)
    embeddings = np.asarray(embeddings).astype("float32")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, docs, embed_model


def query_faiss(query: str, index: Any, docs: List[Dict[str, Any]], embed_model: Any, top_k: int = 3) -> List[Dict[str, Any]]:
    if embed_model is None:
        raise RuntimeError("embed_model is required for FAISS queries")
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    q_emb = np.asarray(q_emb).astype("float32")
    if q_emb.ndim == 1:
        q_emb = np.expand_dims(q_emb, 0)
    _, I = index.search(q_emb, top_k)
    results = []
    for idx in I[0]:
        if 0 <= idx < len(docs):
            results.append(docs[idx])
    return results


def answer_with_model(query: str, index: List[Dict[str, Any]], model_client) -> str:
    top = query_index(query, index, top_k=3)
    context = "\n\n".join([t.get("text", "") for t in top])
    prompt = f"Context:\n{context}\n---\nQuestion: {query}\nAnswer concisely:"
    resp = model_client.generate(prompt)
    if isinstance(resp, dict):
        return str(resp.get("text") or resp)
    return str(resp)


def answer_with_faiss(query: str, faiss_index: Any, docs: List[Dict[str, Any]], embed_model: Any, model_client, top_k: int = 3) -> str:
    top = query_faiss(query, faiss_index, docs, embed_model, top_k=top_k)
    context = "\n\n".join([t.get("text", "") for t in top])
    prompt = f"Context:\n{context}\n---\nQuestion: {query}\nAnswer concisely:"
    resp = model_client.generate(prompt)
    if isinstance(resp, dict):
        return str(resp.get("text") or resp)
    return str(resp)


def build_faiss_from_news(news_items: List[Dict[str, Any]], model_name: str = "all-MiniLM-L6-v2") -> tuple:
    if SentenceTransformer is None or faiss is None:
        raise RuntimeError("Missing dependencies: install sentence-transformers and faiss-cpu")

    texts = [n.get("text", "") for n in news_items]
    embed_model = SentenceTransformer(model_name)
    embeddings = embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    if embeddings.ndim == 1:
        embeddings = np.expand_dims(embeddings, 0)
    embeddings = np.asarray(embeddings).astype("float32")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, news_items, embed_model


def prepare_news_prompt(query: str, news_items: List[Dict[str, Any]], top_k: int = 3) -> str:
    context_items = []
    for n in news_items[:top_k]:
        title = n.get("title", "")
        summary = n.get("summary", "")
        published = n.get("published", "")
        fetched = n.get("fetched_at", "")
        context_items.append(f"- {title} ({published})\n{summary}\n[fetched: {fetched}]")
    context = "\n\n".join(context_items)
    prompt = f"News:\n{context}\n---\nQuestion: {query}\nAnswer:"
    return prompt
