from __future__ import annotations

import asyncio
import io
import logging
import re
from typing import List, Optional, Tuple

from PIL import Image

from utils.config import OllamaConfig
from utils.llm import OllamaChatClient, format_tags

logger = logging.getLogger(__name__)


_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "of",
    "for",
    "with",
    "to",
    "from",
    "by",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "it",
    "this",
    "that",
    "these",
    "those",
    "image",
    "photo",
    "picture",
    "there",
    "here",
    "into",
    "out",
    "over",
    "under",
    "about",
}


def _first_sentence(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    # Common sentence delimiters.
    parts = re.split(r"(?<=[.!?])\s+", text)
    if parts:
        return parts[0].strip()
    return text


class VisionCaptioner:
    def __init__(self, *, model_name: str, llm: OllamaConfig) -> None:
        self.model_name = model_name
        self.llm_client = OllamaChatClient(llm)

        self._captioner = None
        self._lock = asyncio.Lock()

        # Heuristic fallback uses this many tags.
        self._n_tags = 3

    async def caption_and_tags(
        self,
        image_bytes: bytes,
        *,
        telegram_caption: Optional[str] = None,
    ) -> Tuple[str, List[str]]:
        caption = await self._generate_caption(
            image_bytes=image_bytes,
            telegram_caption=telegram_caption,
        )
        tags: List[str] = []

        # Prefer Ollama for consistent tag formatting.
        try:
            tags = await self._generate_tags_with_llm(caption)
        except Exception:
            logger.exception("Tag generation with LLM failed, falling back")
            tags = self._tags_from_caption_heuristic(caption)

        tags = tags[: self._n_tags]
        while len(tags) < self._n_tags:
            tags.append("tag")

        return caption, tags

    async def _generate_caption(
        self,
        *,
        image_bytes: bytes,
        telegram_caption: Optional[str] = None,
    ) -> str:
        # `image-text-to-text` requires a `text` argument (may be "" for unconditional captioning).
        text_for_model = (telegram_caption or "").strip()

        async with self._lock:
            if self._captioner is None:
                await asyncio.to_thread(self._load_captioner)

            img = await asyncio.to_thread(self._bytes_to_pil, image_bytes)
            caption = await asyncio.to_thread(
                self._run_captioner_sync,
                img,
                text_for_model,
            )
            return _first_sentence(caption)

    def _load_captioner(self) -> None:
        from transformers import pipeline

        # For CPU-only environments, `device=-1` is safe.
        try:
            self._captioner = pipeline(
                # Your installed transformers version exposes BLIP via `image-text-to-text`.
                "image-text-to-text",
                model=self.model_name,
                device=-1,
            )
        except Exception:
            # Some environments may require device mapping; rethrow with clarity.
            logger.exception("Failed to load caption model")
            raise

    @staticmethod
    def _bytes_to_pil(image_bytes: bytes) -> Image.Image:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")  # type: ignore[name-defined]
        return img

    def _run_captioner_sync(self, img: Image.Image, text_prompt: str) -> str:
        # Must use keyword args: a single positional is treated as `images` only and leaves `text=None`.
        out = self._captioner(  # type: ignore[operator]
            images=img,
            text=text_prompt,
            max_new_tokens=40,
        )
        if not out:
            return "An image."
        if isinstance(out, list):
            text = out[0].get("generated_text") or out[0].get("text") or ""
        else:
            text = getattr(out, "generated_text", None) or ""
        return (text or "An image.").strip()

    async def _generate_tags_with_llm(self, caption: str) -> List[str]:
        system_prompt = (
            "You are a strict tag generator for image captions. "
            "Return exactly 3 short tags separated by commas. "
            "No extra text."
        )
        user_prompt = (
            f'Caption: "{caption}"\n\n'
            "Tags (exactly 3, comma-separated):"
        )

        raw = await self.llm_client.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=48,
            temperature=0.1,
        )

        # Normalize: allow newlines, hashes, and extra spaces.
        parts = [p.strip() for p in raw.replace("\n", ",").split(",")]
        parts = [p for p in parts if p]
        # Ensure exactly 3.
        formatted = format_tags(parts).split(", ")
        return [t.strip() for t in formatted if t.strip()][:3]

    def _tags_from_caption_heuristic(self, caption: str) -> List[str]:
        # Basic keyword extraction fallback.
        words = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", caption.lower())
        freq = {}
        for w in words:
            w2 = w.strip("_-")
            if not w2 or w2 in _STOPWORDS:
                continue
            if len(w2) < 3:
                continue
            freq[w2] = freq.get(w2, 0) + 1

        ranked = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
        tags: List[str] = []
        seen = set()
        for w, _count in ranked:
            if w in seen:
                continue
            tags.append(w)
            seen.add(w)
            if len(tags) >= self._n_tags:
                break
        return tags

