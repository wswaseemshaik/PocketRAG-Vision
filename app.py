import logging
import shutil
import subprocess
from typing import Set

import requests

from bot.handlers import register_handlers
from rag.rag_pipeline import MiniRAGPipeline
from vision.caption import VisionCaptioner
from utils.config import Settings

from telegram.ext import Application


def _ollama_available_models(base_url: str, timeout_s: float = 5.0) -> Set[str]:
    url = f"{base_url.rstrip('/')}/api/tags"
    resp = requests.get(url, timeout=timeout_s)
    resp.raise_for_status()
    data = resp.json()
    models = data.get("models", [])
    names: Set[str] = set()
    for item in models:
        name = (item.get("name") or item.get("model") or "").strip()
        if name:
            names.add(name)
    return names


def _model_name_candidates(model_name: str) -> Set[str]:
    model_name = model_name.strip()
    if not model_name:
        return set()
    candidates = {model_name}
    if ":" not in model_name:
        candidates.add(f"{model_name}:latest")
    return candidates


def _ensure_ollama_model(settings: Settings, logger: logging.Logger) -> None:
    model_name = settings.llm.model
    base_url = settings.llm.base_url

    try:
        installed = _ollama_available_models(base_url=base_url)
    except Exception as exc:
        logger.warning(
            "Could not query Ollama models at %s (%s). "
            "Skipping auto-download; ensure Ollama is running.",
            base_url,
            exc,
        )
        return

    wanted = _model_name_candidates(model_name)
    if installed.intersection(wanted):
        logger.info("Ollama model `%s` is available", model_name)
        return

    if shutil.which("ollama") is None:
        logger.warning(
            "Model `%s` is not installed in Ollama, but `ollama` CLI is not found in PATH.",
            model_name,
        )
        return

    logger.info("Ollama model `%s` not found. Pulling automatically...", model_name)
    try:
        subprocess.run(
            ["ollama", "pull", model_name],
            check=True,
            text=True,
        )
        logger.info("Finished pulling Ollama model `%s`", model_name)
    except subprocess.CalledProcessError:
        logger.exception("Failed to pull Ollama model `%s`", model_name)


def main() -> None:
    import asyncio

    settings = Settings.from_env()

    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    _ensure_ollama_model(settings=settings, logger=logger)

    application = Application.builder().token(settings.bot_token).build()

    rag = MiniRAGPipeline(
        docs_dir=settings.docs_dir,
        db_path=settings.rag_db_path,
        embedding_model_name=settings.embedding_model_name,
        top_k=settings.rag_top_k,
        llm=settings.llm,
        chunk_size_words=settings.chunk_size_words,
        chunk_overlap_words=settings.chunk_overlap_words,
    )

    vision = VisionCaptioner(
        model_name=settings.vision_model_name,
        llm=settings.llm,
    )

    # Build the index upfront so `/ask` works immediately.
    logger.info("Building RAG index (first run may take a while)...")
    asyncio.run(rag.build_index_if_needed())
    logger.info("RAG index ready")

    application.bot_data["rag"] = rag
    application.bot_data["vision"] = vision

    register_handlers(application=application, rag_pipeline=rag, vision_captioner=vision)

    logger.info("Bot started")

    # python-telegram-bot expects an event loop to be set in the main thread.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    application.run_polling()


if __name__ == "__main__":
    main()

