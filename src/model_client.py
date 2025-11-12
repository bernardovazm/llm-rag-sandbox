from typing import Optional, Dict, Any
import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class OllamaClient:
    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None) -> None:
        self.base_url = (base_url or os.getenv("OLLAMA_URL", "http://localhost:11434")).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", "llama2")
        self.retries = int(os.getenv("OLLAMA_RETRIES", "2"))
        self.backoff = float(os.getenv("OLLAMA_BACKOFF", "0.5"))
        self.timeout = int(os.getenv("OLLAMA_TIMEOUT", "30"))

        self.session = requests.Session()
        retry_strategy = Retry(
            total=self.retries,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST", "GET"],
            backoff_factor=self.backoff,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def generate(self, prompt: str, max_tokens: int = 512, timeout: Optional[int] = None) -> Dict[str, Any]:
        data: Any = None
        try:
            import ollama as _ollama
            if hasattr(_ollama, "generate"):
                data = _ollama.generate(model=self.model, prompt=prompt, max_tokens=max_tokens)
            elif hasattr(_ollama, "completion") and hasattr(_ollama.completion, "create"):
                data = _ollama.completion.create(model=self.model, prompt=prompt, max_tokens=max_tokens)
            elif hasattr(_ollama, "chat") and hasattr(_ollama.chat, "completion"):
                data = _ollama.chat.completion.create(model=self.model, prompt=prompt, max_tokens=max_tokens)
        except Exception:
            data = None

        if data is None:
            url = f"{self.base_url}/api/generate"
            payload = {"model": self.model, "prompt": prompt, "max_tokens": max_tokens}
            req_timeout = timeout or self.timeout
            resp = self.session.post(url, json=payload, timeout=req_timeout)
            resp.raise_for_status()
            data = resp.json()

        if isinstance(data, dict):
            if "text" in data:
                return {"text": data.get("text")}
            gen = data.get("generations") or data.get("choices") or data.get("results")
            if isinstance(gen, list) and len(gen) > 0:
                first = gen[0]
                if isinstance(first, dict):
                    text = first.get("text") or first.get("content") or first.get("output")
                    if text:
                        return {"text": text}
            return data

        return {"text": str(data)}

