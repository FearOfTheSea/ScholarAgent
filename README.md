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
- Hierarchical document summaries, structured quizzes, and flashcards
- A local FastAPI API and Streamlit interface
- A bounded, persistent **Study Mission** that plans prerequisite-valid objectives,
  teaches, assesses, remediates, and resumes for one selected PDF
- A stateless **Quick Ask** compatibility route using the same cited capabilities
- An optional localized ScholarGPT checkpoint: a custom GPT-2 124M architecture
  with response-only instruction tuning, cached decoding, grounded extraction,
  and contract-safe structured generation
- Persistent adaptive tutoring with cited document maps, Socratic activities,
  answer assessment, deterministic mastery tracking, and resumable local sessions
- Verifiable Mission Intelligence with a bounded chained ledger, deterministic
  learning signals, tamper-aware verification, and redacted local export
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
| `LLM_PROVIDER_TYPE` | `ollama` | `ollama` or the local `scratch_gpt` adapter |
| `MODEL_NAME` | `qwen3:1.7b` | Locally installed Ollama model |
| `SCRATCH_GPT_CHECKPOINT_PATH` | `./data/scholar_gpt.pt` | Trained custom-GPT checkpoint |
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
| `POST` | `/agent/requests` | Route one free-form, single-PDF study request |
| `POST` | `/agent/sessions` | Build a cited learning map and start an adaptive session |
| `GET` | `/agent/sessions` | List resumable missions with document/status filters |
| `POST` | `/agent/sessions/{session_id}/turns` | Submit one learner turn |
| `POST` | `/agent/sessions/{session_id}/advance` | Continue or submit a learner response |
| `POST` | `/agent/sessions/{session_id}/complete` | Mark a mission complete with a recap |
| `GET` | `/agent/sessions/{session_id}/insights` | Read deterministic Mission Intelligence signals |
| `GET` | `/agent/sessions/{session_id}/record` | Export a redacted, versioned mission record |
| `POST` | `/agent/sessions/{session_id}/record/verify` | Verify the mission ledger |
| `GET` | `/agent/sessions/{session_id}` | Resume complete local tutor state |
| `DELETE` | `/agent/sessions/{session_id}` | Delete a tutor session but retain its PDF |
| `POST` | `/agent/study` | Deprecated compatibility route for the study agent |
| `POST` | `/questions` | Ask a cited question over one selected document |
| `POST` | `/documents/{document_id}/summary` | Create a local hierarchical summary |
| `POST` | `/documents/{document_id}/quiz` | Create validated quiz questions |
| `POST` | `/documents/{document_id}/flashcards` | Create validated flashcards |

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

Run the real local-model adaptive-tutor journey against the bundled
linear-regression PDF on Windows with:

```powershell
$env:RUN_LOCAL_E2E = "1"
.\.venv\Scripts\python.exe -m pytest tests/test_adaptive_tutor_e2e.py -vv
```

This opt-in test uses the selected local LLM provider, BGE-M3, FAISS, SQLite,
PyMuPDF, FastAPI, and the real example document. It ingests the PDF, creates a
session, explains a topic, assesses an answer, resumes the session, deletes the
document, and verifies session cleanup. Set `E2E_LLM_PROVIDER=scratch_gpt` to
run it through ScholarGPT.

The unified Study Agent has a separate all-capabilities journey:

```powershell
$env:RUN_LOCAL_E2E = "1"
.\.venv\Scripts\python.exe -m pytest tests/test_study_agent_e2e.py -vv
```

Run the same real-PDF journey through ScholarGPT with:

```powershell
$env:RUN_LOCAL_E2E = "1"
$env:E2E_LLM_PROVIDER = "scratch_gpt"
.\.venv\Scripts\python.exe -m pytest tests/test_study_agent_e2e.py -vv
```

ScholarGPT can be trained and evaluated entirely through the project `.venv`
after the GPT-2 files have been cached:

```powershell
.\.venv\Scripts\python.exe scripts\train_scholar_gpt.py --offline
.\.venv\Scripts\python.exe scripts\evaluate_scholar_gpt.py `
  --checkpoint data\scholar_gpt.pt --require-score 6
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

The `POST /agent/requests` endpoint and **Quick Ask** Streamlit page accept
one PDF and a request such as “Explain gradient descent,” “Create 50 quiz
questions,” or “Prepare me for an exam.” Questions use semantic search, the
cited document map, and an explanation; material requests use cited summary,
quiz, and flashcard capabilities. The selected document is injected by the
application and is never chosen by the model. Direct question, summary, quiz,
and flashcard endpoints remain available.

## Study Mission demonstration

The **Study Mission** Streamlit page turns one selected PDF and learner goal into
a persistent learning workspace. Starting a mission creates and caches a cited
document map, selects prerequisite-valid objectives for guided, exam, or cram
mode, and displays the ordered plan. The mission then explains concepts, asks
short-answer questions, gives progressive hints, assesses learner responses on
a validated 0–3 rubric, and inserts cited remediation when needed.

Every generated material and learner-facing mission response carries validated
page/chunk evidence from the selected document. Mastery is calculated in
application code from scored attempts; the model cannot assign or mutate mastery
directly. Sessions, artifacts, and concise capability traces are stored locally
in SQLite and survive application restarts. Optional artifact failures remain
visible without discarding the resumable mission.
