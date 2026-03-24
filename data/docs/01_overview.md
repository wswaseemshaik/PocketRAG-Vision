# Mini-RAG Telegram Bot (Overview)

This project demonstrates a small retrieval-augmented generation (Mini-RAG) system for a Telegram bot.

The bot supports commands like `/ask <query>` and answers questions by:
1. Loading local text/markdown documents from `data/docs/`.
2. Splitting the documents into overlapping chunks.
3. Embedding each chunk using a sentence-transformers model (`all-MiniLM-L6-v2` by default).
4. Storing embeddings in SQLite so retrieval stays fast.
5. At query time, embedding the user question and retrieving the top matching chunks by cosine similarity.
6. Sending the retrieved context to a local LLM (Ollama is preferred) to generate a concise answer.

The goal is to keep the implementation simple, modular, and easy to run locally without external APIs.

If the local LLM is unavailable, the bot will still run but will return a fallback message for `/ask`.

