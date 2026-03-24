# Local models

## `blip-image-captioning-base/`

Bundled **BLIP** weights for `/image` captioning:

- `pytorch_model.bin` — main weights (rename your downloaded `fd40eaea…bin` to this name)
- `config.json`, `preprocessor_config.json`, tokenizer files — from [Hugging Face](https://huggingface.co/Salesforce/blip-image-captioning-base)

Set in `.env`:

```env
VISION_MODEL_NAME="models/blip-image-captioning-base"
```

The app resolves this path relative to the project root.
