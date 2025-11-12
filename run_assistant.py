import argparse
from typing import Any, Dict, Optional, Tuple

from src.rag_pipeline import (
    build_index,
    answer_with_model,
    build_faiss_index,
    answer_with_faiss,
)
from src.rag_pipeline import prepare_news_prompt
from src.model_client import OllamaClient
from src.langchain_integration import build_langchain_faiss, query_langchain
from src.local_model_client import LocalModelClient
from src.news_fetcher import fetch_latest_news
from src.rag_pipeline import build_faiss_from_news


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--no-index", action="store_true")
    parser.add_argument(
        "--use-langchain",
        action="store_true",
        help="Use LangChain embeddings + FAISS vectorstore for retrieval",
    )
    parser.add_argument(
        "--use-faiss", action="store_true"
    )
    parser.add_argument("--prefer", choices=["heavy", "light"], default=None)
    parser.add_argument("--force-heavy", action="store_true")
    parser.add_argument("--offload-folder", type=str, default=None)
    parser.add_argument("--refresh-news", action="store_true")
    parser.add_argument("--news-sources", type=str, default=None)
    parser.add_argument("--news-limit", type=int, default=5)
    parser.add_argument("--wiki-fallback", action="store_true")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--num-return-sequences", type=int, default=1)
    return parser.parse_args()


def init_components(args: argparse.Namespace) -> Tuple[OllamaClient, Dict[str, Any]]:
    client_impl: Any
    if args.backend == "ollama":
        client_impl = OllamaClient()
    else:
        prefer = args.prefer if getattr(args, "prefer", None) is not None else None
        force_heavy = getattr(args, "force_heavy", False)
        offload_folder = getattr(args, "offload_folder", None)
        client_impl = LocalModelClient(prefer=prefer, force_heavy=force_heavy, offload_folder=offload_folder)
    components: Dict[str, Any] = {"vectorstore": None, "faiss_index": None, "faiss_docs": None, "embed_model": None, "index": []}

    if args.use_langchain and not args.no_index:
        components["vectorstore"], components["embed_model"] = build_langchain_faiss(args.root)
    elif args.use_faiss and not args.no_index:
        components["faiss_index"], components["faiss_docs"], components["embed_model"] = build_faiss_index(args.root)
    elif args.refresh_news:
        sources = None
        if getattr(args, "news_sources", None):
            sources = [s.strip() for s in args.news_sources.split(",") if s.strip()]
        news_items = fetch_latest_news(sources=sources, limit_per_source=getattr(args, "news_limit", 5))
        if not news_items and getattr(args, "wiki_fallback", False):
            from src.news_fetcher import fetch_wikipedia_year

            news_items = fetch_wikipedia_year()

        if news_items:
            components["faiss_index"], components["faiss_docs"], components["embed_model"] = build_faiss_from_news(news_items)
            components["news_items"] = news_items
    elif not args.no_index:
        components["index"] = build_index(args.root)

    return client_impl, components


def handle_query(
    q: str, client: OllamaClient, components: Dict[str, Any], args: argparse.Namespace
) -> None:
    gen_kwargs: Dict[str, Any] = {}
    if getattr(args, "temperature", None) is not None:
        gen_kwargs["temperature"] = args.temperature
    if getattr(args, "do_sample", False):
        gen_kwargs["do_sample"] = True
    if getattr(args, "max_new_tokens", None) is not None:
        gen_kwargs["max_new_tokens"] = args.max_new_tokens
    if getattr(args, "top_k", None) is not None:
        gen_kwargs["top_k"] = args.top_k
    if getattr(args, "top_p", None) is not None:
        gen_kwargs["top_p"] = args.top_p
    if getattr(args, "num_return_sequences", None) is not None:
        gen_kwargs["num_return_sequences"] = args.num_return_sequences

    if args.no_index:
        _handle_no_index(q, client, gen_kwargs)
        return

    if components.get("vectorstore") is not None:
        _handle_vectorstore(q, client, components, gen_kwargs)
        return

    if components.get("faiss_index") is not None:
        _handle_faiss(q, client, components, gen_kwargs)
        return

    print(answer_with_model(q, components.get("index", []), client))


def _handle_no_index(q: str, client: OllamaClient, gen_kwargs: Dict[str, Any]) -> None:
    prompt = f"Question: {q}\nAnswer concisely:"
    print(client.generate(prompt, **gen_kwargs))


def _handle_vectorstore(q: str, client: OllamaClient, components: Dict[str, Any], gen_kwargs: Dict[str, Any]) -> None:
    docs = query_langchain(components["vectorstore"], q, k=3)
    context = "\n\n".join([d.page_content for d in docs])
    prompt = f"Context:\n{context}\n---\nQuestion: {q}\nAnswer concisely:"
    print(client.generate(prompt, **gen_kwargs))


def _handle_faiss(q: str, client: OllamaClient, components: Dict[str, Any], gen_kwargs: Dict[str, Any]) -> None:
    if components.get("news_items"):
        news = components["news_items"]
        try:
            print(f"\nUsing {len(news)} news items for context (showing up to 5):\n")
            for i, n in enumerate(news[:5], start=1):
                title = n.get("title", "")
                pub = n.get("published", "")
                fetched = n.get("fetched_at", "")
                summary = (n.get("summary") or "").strip()
                if len(summary) > 400:
                    summary = summary[:400].rsplit(" ", 1)[0] + "..."
                link = n.get("link", "")
                print(f"{i}. {title} ({pub}) [fetched: {fetched}]\n   {summary}\n   link: {link}\n")
        except Exception:
            pass

        prompt = prepare_news_prompt(q, news, top_k=3)
        print("\n--- Prompt sent to model (preview) ---\n")
        print(prompt)
        print("\n--- Model output ---\n")
        resp = client.generate(prompt, **gen_kwargs)
        try:
            if isinstance(resp, dict):
                out = resp.get("text") or resp.get("generated_text") or str(resp)
            else:
                out = str(resp)
        except Exception:
            out = str(resp)
        print(out)
        return

    print(answer_with_faiss(q, components["faiss_index"], components["faiss_docs"], components["embed_model"], client, top_k=3, **gen_kwargs))


def main() -> None:
    args = parse_args()
    client, components = init_components(args)

    try:
        while True:
            q = input("query> ").strip()
            if not q:
                continue
            if q.lower() in ("exit", "quit"):
                break
            handle_query(q, client, components, args)
    except (KeyboardInterrupt, EOFError):
        print("\nbye")


if __name__ == "__main__":
    main()
