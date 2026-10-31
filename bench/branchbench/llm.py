"""A minimal OpenAI-compatible chat client (standard library only) and a counting fake.

Configuration comes from environment variables only; nothing is written to disk except
the model name and base URL (never the key):

    BB_LLM_BASE_URL   e.g. https://api.deepseek.com/v1
    BB_LLM_API_KEY    the key (read from the environment, never logged)
    BB_LLM_MODEL      e.g. deepseek-chat

Every call is made with temperature 0. ``usage`` accumulates the token counts that the
provider reports, so a run's manifest carries its real cost basis.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List

Messages = List[Dict[str, str]]


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prompt_chars: int = 0
    completion_chars: int = 0
    by_purpose: Dict[str, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, purpose: str, messages: Messages, text: str, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        with self._lock:
            self.calls += 1
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            self.prompt_chars += sum(len(m.get("content") or "") for m in messages)
            self.completion_chars += len(text)
            self.by_purpose[purpose] = self.by_purpose.get(purpose, 0) + 1

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


class LLMError(RuntimeError):
    pass


class ChatClient:
    def __init__(self, *, base_url: str, api_key: str, model: str, timeout: float = 120.0,
                 max_retries: int = 5, max_tokens: int = 1024) -> None:
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.usage = Usage()

    @classmethod
    def from_env(cls) -> "ChatClient":
        missing = [name for name in ("BB_LLM_BASE_URL", "BB_LLM_API_KEY", "BB_LLM_MODEL") if not os.environ.get(name)]
        if missing:
            raise LLMError(f"missing environment variables: {', '.join(missing)}")
        return cls(base_url=os.environ["BB_LLM_BASE_URL"], api_key=os.environ["BB_LLM_API_KEY"],
                   model=os.environ["BB_LLM_MODEL"])

    def describe(self) -> dict:
        return {"kind": "openai_compatible", "base_url": self.base_url, "model": self.model,
                "temperature": 0, "max_tokens": self.max_tokens}

    def __call__(self, messages: Messages, *, purpose: str = "other") -> str:
        body = json.dumps({"model": self.model, "messages": messages, "temperature": 0,
                           "max_tokens": self.max_tokens, "stream": False}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body, method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self._api_key}"},
        )
        delay = 2.0
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                text = data["choices"][0]["message"]["content"] or ""
                usage = data.get("usage") or {}
                self.usage.add(purpose, messages, text, int(usage.get("prompt_tokens") or 0),
                               int(usage.get("completion_tokens") or 0))
                return text
            except urllib.error.HTTPError as exc:
                retryable = exc.code in (408, 409, 429) or exc.code >= 500
                if not retryable or attempt == self.max_retries:
                    raise LLMError(f"HTTP {exc.code}: {exc.read()[:300]!r}") from None
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                if attempt == self.max_retries:
                    raise LLMError(f"network error: {type(exc).__name__}") from None
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
        raise LLMError("unreachable")


class CountingFake:
    """Calls nothing. Records the prompts it would have sent and answers with a canned reply.

    Used by ``estimate`` and by the dry run: the extractor gets a JSON note that cites
    every exchange, summaries get the last new message, answers get the empty string.
    """

    def __init__(self) -> None:
        self.model = "counting-fake"
        self.usage = Usage()

    def describe(self) -> dict:
        return {"kind": "counting_fake"}

    def __call__(self, messages: Messages, *, purpose: str = "other") -> str:
        if purpose == "extract":
            numbers = [line.strip("[]") for line in messages[-1]["content"].splitlines() if line.startswith("[")]
            text = json.dumps({"notes": [{"kind": "fact", "text": "x" * 40,
                                          "exchanges": [int(n) for n in numbers if n.isdigit()]}]})
        elif purpose == "summary":
            text = "摘" * 200
        else:
            text = "答" * 30
        self.usage.add(purpose, messages, text)
        return text
