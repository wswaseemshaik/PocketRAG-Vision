# Chunking and Embeddings

Retrieval quality often depends on how documents are chunked before embedding.

In this Mini-RAG pipeline, documents are split using a simple word-based strategy:

- `CHUNK_SIZE_WORDS` controls the maximum number of words per chunk.
- `CHUNK_OVERLAP_WORDS` provides overlap between adjacent chunks so important information isn't lost at boundaries.

Each chunk is embedded with `sentence-transformers/all-MiniLM-L6-v2` by default.
Embeddings are stored in a SQLite database (`data/embeddings.sqlite3`) along with:

- chunk text
- source document path
- chunk index within the document

During retrieval, the bot embeds the user query and compares it against all chunk embeddings using cosine similarity.
It then selects the top `RAG_TOP_K` chunks to include in the prompt.

