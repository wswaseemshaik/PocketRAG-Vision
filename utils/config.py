from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv(dotenv_path: Path) -> None:
    """
    Minimal .env loader to avoid extra dependencies.
    Supports `KEY=value` and ignores comments/blank lines.
    """
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass(frozen=True)
class OllamaConfig:
    base_url: str
    model: str
    temperature: float
    num_predict: int


@dataclass(frozen=True)
class Settings:
    bot_token: str
    log_level: str

    docs_dir: Path
    rag_db_path: Path

    embedding_model_name: str
    vision_model_name: str

    chunk_size_words: int
    chunk_overlap_words: int
    rag_top_k: int

    llm: OllamaConfig

    @staticmethod
    def from_env() -> "Settings":
        base_dir = Path(__file__).resolve().parents[1]
        _load_dotenv(base_dir / ".env")

        bot_token = os.environ.get("BOT_TOKEN") or ""
        if not bot_token:
            raise RuntimeError(
                "BOT_TOKEN is missing. Add it to the .env file at the project root."
            )

        log_level = os.environ.get("LOG_LEVEL", "INFO")

        docs_dir = base_dir / "data" / "docs"
        rag_db_path = base_dir / "data" / "embeddings.sqlite3"

        embedding_model_name = os.environ.get(
            "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
        )
        vision_raw = os.environ.get(
            "VISION_MODEL_NAME", "Salesforce/blip-image-captioning-base"
        )
        # If VISION_MODEL_NAME is a real directory under the project (e.g. models/...), use
        # absolute path. Otherwise keep as Hugging Face hub id (e.g. org/model).
        vision_path = Path(vision_raw).expanduser()
        if vision_path.is_absolute():
            vision_model_name = str(vision_path.resolve())
        else:
            local_dir = (base_dir / vision_path).resolve()
            vision_model_name = (
                str(local_dir) if local_dir.is_dir() else vision_raw
            )

        chunk_size_words = int(os.environ.get("CHUNK_SIZE_WORDS", "180"))
        chunk_overlap_words = int(os.environ.get("CHUNK_OVERLAP_WORDS", "40"))
        rag_top_k = int(os.environ.get("RAG_TOP_K", "3"))

        # Ollama (local LLM) for both answering and generating tags from captions.
        llm = OllamaConfig(
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            model=os.environ.get("OLLAMA_MODEL", "llama3"),
            temperature=float(os.environ.get("OLLAMA_TEMPERATURE", "0.2")),
            num_predict=int(os.environ.get("OLLAMA_NUM_PREDICT", "256")),
        )

        return Settings(
            bot_token=bot_token,
            log_level=log_level,
            docs_dir=docs_dir,
            rag_db_path=rag_db_path,
            embedding_model_name=embedding_model_name,
            vision_model_name=vision_model_name,
            chunk_size_words=chunk_size_words,
            chunk_overlap_words=chunk_overlap_words,
            rag_top_k=rag_top_k,
            llm=llm,
        )

