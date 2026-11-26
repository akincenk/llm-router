"""Minimal OpenAI-compatible client (standard library): chat completions and embeddings."""

import json
import time
import urllib.request
from dataclasses import dataclass


@dataclass
class Completion:
    text: str
    tokens: int
    seconds: float
    tokens_in: int = 0  # prompt tokens
    tokens_out: int = 0  # completion tokens


class Client:
    def __init__(self, base_url: str, api_key: str = "local", timeout: float = 300):
        self.base, self.key, self.timeout = base_url.rstrip("/"), api_key, timeout

    def _post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(
            self.base + path,
            data=json.dumps(body).encode(),
            method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.load(r)

    def chat(self, model: str, messages: list[dict], max_tokens: int = 2048) -> Completion:
        start = time.monotonic()
        body = {"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens}
        data = self._post("/chat/completions", body)
        usage = data.get("usage") or {}
        return Completion(
            data["choices"][0]["message"].get("content") or "",
            usage.get("total_tokens", 0),
            time.monotonic() - start,
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
        )

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        data = self._post("/embeddings", {"model": model, "input": texts})
        return [d["embedding"] for d in sorted(data["data"], key=lambda d: d["index"])]
