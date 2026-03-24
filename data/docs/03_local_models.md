# Local Models: Ollama + BLIP

This project prefers local models to keep costs low and avoid external APIs.

## Ollama (LLM)

For answering questions and generating tags, the bot calls Ollama's HTTP API at `http://localhost:11434`.

You can control:
- `OLLAMA_BASE_URL`
- `OLLAMA_MODEL`
- `OLLAMA_TEMPERATURE`
- `OLLAMA_NUM_PREDICT`

Make sure Ollama is running locally and the selected model is pulled.

## Image captioning (BLIP)

For the `/image` command, the bot uses a HuggingFace `transformers` image-to-text model.
By default it uses `Salesforce/blip-image-captioning-base`.

When you upload a photo, the bot generates:
- a one-sentence caption
- exactly three tags (generated via Ollama, with a fallback heuristic if needed)

