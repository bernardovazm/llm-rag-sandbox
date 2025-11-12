Local models RAG and retrieval

Run

- create venv
- install dependencies

```bash
pip install -r requirements.txt
export PYTHONPATH="$(pwd)/llm-rag-sadbox"
python -c "from src.news_fetcher import fetch_wikipedia_events; import datetime; print(fetch_wikipedia_events(datetime.datetime.utcnow().year)[:5])"
python run_assistant.py --backend local --prefer heavy --force-heavy --wiki-fallback
```

# llm-rag-sandbox
