from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

from utils.config import OllamaConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


class OllamaChatClient:
    def __init__(self, cfg: OllamaConfig, timeout_s: float = 120.0) -> None:
        self.cfg = cfg
        self.timeout_s = timeout_s
        # Detect once and reuse the compatible endpoint for lower latency.
        self._chat_endpoint_supported: Optional[bool] = None

    def _post_chat(
        self,
        messages: List[ChatMessage],
        *,
        num_predict: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        url = f"{self.cfg.base_url.rstrip('/')}/api/chat"
        payload: Dict[str, Any] = {
            "model": self.cfg.model,
            "stream": False,
            "options": {
                "temperature": self.cfg.temperature if temperature is None else temperature,
                "num_predict": self.cfg.num_predict if num_predict is None else num_predict,
            },
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        resp = requests.post(url, json=payload, timeout=self.timeout_s)
        resp.raise_for_status()
        data = resp.json()
        content = (
            data.get("message", {}).get("content")
            or data.get("response")
            or data.get("content")
            or ""
        )
        return content.strip()

    def _post_generate(
        self,
        prompt: str,
        *,
        num_predict: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        url = f"{self.cfg.base_url.rstrip('/')}/api/generate"
        payload: Dict[str, Any] = {
            "model": self.cfg.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.cfg.temperature if temperature is None else temperature,
                "num_predict": self.cfg.num_predict if num_predict is None else num_predict,
            },
        }
        resp = requests.post(url, json=payload, timeout=self.timeout_s)
        resp.raise_for_status()
        data = resp.json()
        content = data.get("response") or data.get("content") or ""
        return str(content).strip()

    async def chat(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        extra_messages: Optional[List[ChatMessage]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        messages: List[ChatMessage] = []
        if system_prompt:
            messages.append(ChatMessage(role="system", content=system_prompt))
        if extra_messages:
            messages.extend(extra_messages)
        messages.append(ChatMessage(role="user", content=user_prompt))

        prompt_parts: List[str] = []
        if system_prompt:
            prompt_parts.append(f"System: {system_prompt}")
        if extra_messages:
            # Flatten extra messages into a simple chat transcript.
            for m in extra_messages:
                prompt_parts.append(f"{m.role.capitalize()}: {m.content}")
        prompt_parts.append(f"User: {user_prompt}")
        prompt_parts.append("Assistant:")
        prompt = "\n".join(prompt_parts)

        # Known older Ollama servers: use generate directly, no chat probe each call.
        if self._chat_endpoint_supported is False:
            return await asyncio.to_thread(
                self._post_generate,
                prompt,
                num_predict=max_tokens,
                temperature=temperature,
            )

        try:
            output = await asyncio.to_thread(
                self._post_chat,
                messages,
                num_predict=max_tokens,
                temperature=temperature,
            )
            self._chat_endpoint_supported = True
            return output
        except requests.exceptions.HTTPError as e:
            status_code = getattr(e.response, "status_code", None)
            # Some Ollama versions don't expose `/api/chat` (404). Fall back to `/api/generate`.
            if status_code == 404:
                self._chat_endpoint_supported = False
                return await asyncio.to_thread(
                    self._post_generate,
                    prompt,
                    num_predict=max_tokens,
                    temperature=temperature,
                )
            logger.exception("Ollama chat failed (HTTP %s)", status_code)
            raise
        except Exception:
            logger.exception("Ollama chat failed")
            raise


def format_tags(tags: List[str]) -> str:
    cleaned = []
    for t in tags:
        t2 = t.strip().lstrip("#").strip()
        if not t2:
            continue
        # Keep tags short and safe for Telegram.
        cleaned.append(t2[:32])
    cleaned = cleaned[:3]
    while len(cleaned) < 3:
        cleaned.append("tag")
    return ", ".join(cleaned)

