from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Set

from telegram import Message, Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from rag.rag_pipeline import MiniRAGPipeline
from vision.caption import VisionCaptioner

logger = logging.getLogger(__name__)

# In-memory state: last 3 messages per user.
# Each entry: {"role": "user"|"assistant", "content": "..."}
USER_MEMORY: Dict[int, List[Dict[str, str]]] = defaultdict(list)

# Users who invoked `/image` and are expected to upload one image next.
PENDING_IMAGE: Set[int] = set()

START_TEXT = (
    "Welcome to the Mini-RAG Vision Bot.\n\n"
    "Available commands:\n"
    "/ask <query> - Ask a question from indexed documents\n"
    "/summarize [topic] - Summarize documents (optionally focused)\n"
    "/image - Upload a photo to get caption + 3 tags\n"
    "/help - Show full usage"
)

HELP_TEXT = (
    "Commands:\n"
    "/start - Greeting\n"
    "/help - Usage instructions\n"
    "/ask <query> - Answer using Mini-RAG\n"
    "/summarize [topic] - Summarize indexed documents (optional focus topic)\n"
    "/image - Upload a photo (optional caption on the photo); I return a caption + 3 tags"
)


def _trim_memory(mem: List[Dict[str, str]], max_items: int = 3) -> None:
    while len(mem) > max_items:
        mem.pop(0)


def _remember(user_id: int, role: str, content: str) -> None:
    USER_MEMORY[user_id].append({"role": role, "content": content})
    _trim_memory(USER_MEMORY[user_id], 3)


def _get_message(update: Update) -> Optional[Message]:
    return update.message


def _get_rag(context: ContextTypes.DEFAULT_TYPE) -> MiniRAGPipeline:
    return context.application.bot_data["rag_pipeline"]


def _get_vision(context: ContextTypes.DEFAULT_TYPE) -> VisionCaptioner:
    return context.application.bot_data["vision_captioner"]


async def _download_photo_bytes(
    *,
    message: Message,
    context: ContextTypes.DEFAULT_TYPE,
) -> Optional[bytes]:
    if not message.photo:
        return None

    photo = message.photo[-1]  # highest resolution
    file_id = photo.file_id
    try:
        file = await context.bot.get_file(file_id)
        data = await file.download_as_bytearray()
        return bytes(data)
    except Exception:
        logger.exception("Failed to download image")
        return None


async def _respond_with_caption(
    *,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    image_bytes: bytes,
    telegram_caption: Optional[str],
) -> None:
    message = _get_message(update)
    if message is None:
        return

    user_id = update.effective_user.id
    captioner = _get_vision(context)

    await message.reply_text("Captioning...")
    try:
        caption, tags = await captioner.caption_and_tags(
            image_bytes=image_bytes,
            telegram_caption=telegram_caption,
        )
    except Exception:
        logger.exception("Captioning failed")
        await message.reply_text("Sorry, I couldn't caption that image.")
        PENDING_IMAGE.discard(user_id)
        return

    response = f"Caption: {caption}\nTags: {', '.join(tags)}"
    _remember(user_id, role="assistant", content=response)
    PENDING_IMAGE.discard(user_id)
    await message.reply_text(response)


def register_handlers(
    *,
    application: Application,
    rag_pipeline: MiniRAGPipeline,
    vision_captioner: VisionCaptioner,
) -> None:
    # Command handlers (cleanly separated command entry points).
    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("help", cmd_help))
    application.add_handler(CommandHandler("ask", cmd_ask))
    application.add_handler(CommandHandler("summarize", cmd_summarize))
    application.add_handler(CommandHandler("image", cmd_image))

    # Media handlers (/image upload flow only).
    application.add_handler(MessageHandler(filters.PHOTO & ~filters.COMMAND, on_photo_message))

    application.add_error_handler(error_handler)

    # Store the pipelines on app.bot_data to access inside handler closures.
    application.bot_data["rag_pipeline"] = rag_pipeline
    application.bot_data["vision_captioner"] = vision_captioner


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:  # noqa: ARG001
    message = _get_message(update)
    if message is not None:
        await message.reply_text(START_TEXT)


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:  # noqa: ARG001
    message = _get_message(update)
    if message is not None:
        await message.reply_text(HELP_TEXT)


async def cmd_ask(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = _get_message(update)
    if message is None:
        return

    rag = _get_rag(context)
    user_id = update.effective_user.id

    query = " ".join(context.args).strip()
    if not query:
        await message.reply_text("Usage: `/ask <query>`")
        return

    # Remember the user query for prompt context.
    memory = USER_MEMORY[user_id]
    _remember(user_id, role="user", content=query)

    await message.reply_text("Thinking...")
    try:
        answer = await rag.answer(query=query, memory=memory)
    except Exception:
        logger.exception("RAG answer failed")
        answer = "Sorry, something went wrong while answering your question."

    _remember(user_id, role="assistant", content=answer)
    await message.reply_text(answer)


async def cmd_summarize(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = _get_message(update)
    if message is None:
        return

    rag = _get_rag(context)
    user_id = update.effective_user.id
    topic = " ".join(context.args).strip() or None

    await message.reply_text("Summarizing...")
    try:
        summary = await rag.summarize(query=topic)
    except Exception:
        logger.exception("RAG summarize failed")
        summary = "Sorry, something went wrong while summarizing documents."

    _remember(user_id, role="assistant", content=summary)
    await message.reply_text(summary)


async def cmd_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:  # noqa: ARG001
    message = _get_message(update)
    if message is None:
        return

    user_id = update.effective_user.id
    PENDING_IMAGE.add(user_id)
    USER_MEMORY[user_id] = USER_MEMORY.get(user_id, [])
    await message.reply_text(
        "Send me a photo (optional caption on the photo is passed to the model). "
        "I'll return a caption + 3 tags."
    )


async def on_photo_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = _get_message(update)
    if message is None:
        return

    user_id = update.effective_user.id
    if user_id not in PENDING_IMAGE:
        return

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    image_bytes = await _download_photo_bytes(message=message, context=context)
    if image_bytes is None:
        await message.reply_text("I couldn't download that image. Try again.")
        PENDING_IMAGE.discard(user_id)
        return

    telegram_caption = (message.caption or "").strip() or None
    await _respond_with_caption(
        update=update,
        context=context,
        image_bytes=image_bytes,
        telegram_caption=telegram_caption,
    )


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:  # noqa: ANN401
    logger.exception("Unhandled exception: %s", context.error)
    if isinstance(update, Update) and update.effective_message:
        try:
            await update.effective_message.reply_text("Something went wrong. Try again.")
        except Exception:
            pass

