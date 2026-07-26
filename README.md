# ScholarAgent

ScholarAgent is a local PDF study assistant built as an educational example of
Clean Architecture and Ports & Adapters. It runs on one computer: Ollama
generates responses locally, BGE-M3 creates embeddings locally, and PDFs,
metadata, and vectors remain on disk under the project data directory.

No paid API, cloud inference service, account, or authentication is required.
The initial model downloads are the only network-dependent setup step; normal
use can continue offline once those files are cached.

## What is implemented

- A persistent local PDF library with upload, listing, and explicit deletion
- PDF extraction with PyMuPDF, page-aware chunking, BGE-M3 embeddings, SQLite
  catalog metadata, and a normalized FAISS index
- Grounded question answering with document, page, chunk, and similarity-score
  citations
- Hierarchical document summaries, structured quizzes, flashcards, and
  evidence-preserving two-document comparisons
- A local FastAPI API and Streamlit interface
- A goal-oriented exam preparation agent that plans and runs several study tools
- Dependency injection that keeps Ollama, FAISS, PyMuPDF, LangChain, and
  LangGraph behind application output ports

The current PDF workflow is designed for English study material. It rejects
empty, malformed, oversized, and non-PDF uploads before loading an embedding
model or calling Ollama.

## Local runtime profile

The safe default for a 4 GB RTX 3050 and 16 GB system memory is:

- Ollama model: `qwen3:1.7b`
- Context length: `4096`
- Concurrent Ollama requests: `1`
- Loaded Ollama models: `1`
- Embeddings: `BAAI/bge-m3` on `cpu`

`qwen3:4b` remains an opt-in benchmark configuration, not the default. Change
`MODEL_NAME` only after checking memory and response time on the target laptop.

## Setup and start

1. Install Python 3.12, [uv](https://docs.astral.sh/uv/), and
   [Ollama](https://ollama.com/).
2. Copy `.env.example` to `.env` and adjust local paths only if needed.
3. Install Python dependencies:

   ```bash
   uv sync --all-groups
   ```

4. Download the default local language model once:

   ```bash
   ollama pull qwen3:1.7b
   ```

5. Start the local API, bound only to `127.0.0.1`:

   ```bash
   uv run scholaragent-api
   ```

6. In another terminal, start the local UI:

   ```bash
   uv run scholaragent-web
   ```

The BGE-M3 files download on the first valid PDF ingestion and are cached by
Sentence Transformers. The application can still start before either model is
available: `GET /health` succeeds, while `GET /ready` reports `unavailable`.

## Configuration

All values are environment variables and have matching entries in
`.env.example`.

| Setting | Default | Purpose |
| --- | --- | --- |
| `MODEL_NAME` | `qwen3:1.7b` | Locally installed Ollama model |
| `OLLAMA_URL` | `http://localhost:11434` | Local Ollama HTTP endpoint |
| `LLM_CONTEXT_LENGTH` | `4096` | Ollama context window |
| `LLM_MAX_TOKENS` | `1024` | Maximum generated tokens |
| `OLLAMA_NUM_PARALLEL` | `1` | Documented local concurrency limit |
| `OLLAMA_MAX_LOADED_MODELS` | `1` | Documented model-residency limit |
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-m3` | Sentence Transformers embedding model |
| `EMBEDDING_DEVICE` | `cpu` | Embedding execution device |
| `DOCUMENT_LIBRARY_PATH` | `./data/documents` | Retained original PDFs |
| `CATALOG_DB_PATH` | `./data/catalog.sqlite3` | SQLite document catalog |
| `VECTOR_DB_PATH` | `./data/vector_store` | FAISS index and chunk metadata |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `120` | Page-aware chunking parameters |
| `RETRIEVAL_TOP_K` | `4` | Evidence chunks retrieved per query |
| `MAX_UPLOAD_MB` | `50` | Per-PDF local upload limit |
| `DEBUG` | `false` | FastAPI debug mode |

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Process liveness; never performs inference |
| `GET` | `/ready` | Checks local Ollama and the configured model without inference |
| `POST` | `/documents` | Upload, retain, and index one PDF (multipart field: `file`) |
| `GET` | `/documents` | List locally indexed documents |
| `DELETE` | `/documents/{document_id}` | Remove the retained PDF, catalog record, chunks, and vectors |
| `POST` | `/questions` | Ask a cited question over selected documents |
| `POST` | `/documents/{document_id}/summary` | Create a local hierarchical summary |
| `POST` | `/documents/{document_id}/quiz` | Create validated quiz questions |
| `POST` | `/documents/{document_id}/flashcards` | Create validated flashcards |
| `POST` | `/comparisons` | Compare two documents with evidence from each |

Interactive API documentation is available at
`http://127.0.0.1:8000/docs` while the API is running.

## Local storage and deletion

```text
data/
  documents/                 retained source PDFs, named by document ID
  catalog.sqlite3            document titles, sources, dates, and page counts
  vector_store/
    index.faiss              normalized inner-product vector index
    metadata.sqlite3         page, chunk, and vector-ID metadata
```

Deleting a document through the API or Streamlit Library removes its source
file, SQLite catalog record, chunk metadata, and FAISS vectors. Nothing is
uploaded to a ScholarAgent service.

## Quality checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
python -m compileall -q src tests
```

## Evaluation and benchmark

The repository includes owner-supplied English lecture PDFs with expected
page-level retrieval citations. Run the deterministic, no-download regression
test with the regular suite, or evaluate the configured BGE-M3 model directly:

```bash
uv run --group dev python scripts/evaluate_retrieval.py
```

After installing Ollama and pulling `qwen3:1.7b`, record the local laptop's
response time and memory behavior with:

```bash
uv run --group dev python scripts/benchmark_local_runtime.py
```

See [Evaluation](docs/Evaluation.md) and [Local Benchmark](docs/Benchmarks.md)
for the corpus contract and recorded results.

See [Architecture](docs/Architecture.md) and the
[Folder Guide](docs/FolderGuide.md) for the dependency boundaries and package
layout.

## Agent demonstration

The `POST /agent/study` endpoint and the **Study Agent** Streamlit page accept a
goal such as “Prepare me for an exam using these lecture PDFs.” The LangGraph
workflow plans an evidence search, document summaries, an optional comparison,
and a structured quiz, then returns recommendations and source citations. The
planner is constrained to the local study tools and preserves partial results
when an optional step is unavailable.
