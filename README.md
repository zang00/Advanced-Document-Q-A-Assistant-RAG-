# Document Q&A Prototype

A small backend-only RAG application: upload PDF/TXT, retrieve relevant chunks from
embedded persistent Chroma, and answer using OpenAI through LangChain. No local ML
models, frontend, separate database service, job queue, or retrieval extensions.

## Run locally

Use Python 3.13 (the Docker image and development checks use this version).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Set OPENAI_API_KEY in .env before uploading or chatting.
uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

PowerShell: activate with `.\.venv\Scripts\Activate.ps1`, copy the configuration
with `Copy-Item .env.example .env`, and use `curl.exe` instead of the `curl` alias.
Interactive API documentation is at <http://localhost:8000/docs>.

## Four endpoints

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Liveness and whether an API key is configured; no paid API call |
| `POST /upload` | One PDF or UTF-8 TXT file per request; returns after indexing |
| `GET /documents` | List indexed documents and chunk counts |
| `POST /chat` | Question and optional session ID; returns an answer and citations |

Examples below use Bash. Upload additional documents with additional requests.

```bash
curl http://localhost:8000/health
printf 'Below -20 C, wear insulated gloves.\n' > safety.txt
curl -X POST http://localhost:8000/upload -F 'file=@safety.txt'
curl -X POST http://localhost:8000/upload -F 'file=@manual.pdf'
curl http://localhost:8000/documents
curl -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"question":"What should I wear below -20 C?","session_id":"demo"}'
curl -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"question":"When should I wear them?","session_id":"demo"}'
```

Illustrative response shape (actual wording depends on the model):

```json
{
  "session_id": "demo",
  "answer": "Wear insulated gloves below -20 C.",
  "citations": [
    {
      "document": "safety.txt",
      "page": null,
      "snippet": "Below -20 C, wear insulated gloves."
    }
  ]
}
```

PDF pages are numbered from 1; TXT citations have `page: null`. A snippet is the
complete retrieved chunk, copied from its stored metadata. The LLM selects source
IDs but cannot supply document names, page numbers, or quotation text. Missing
answers and invalid/missing source references return exactly:

> I couldn't find this in the provided documents.

## How it works

1. Parse PDF page-by-page or decode UTF-8 TXT in `app/documents.py`.
2. Split text with LangChain, preserving document/page metadata.
3. Generate OpenAI API embeddings and save chunks in local Chroma.
4. Retrieve `TOP_K` chunks using the question and preceding session question.
5. Ask the LLM for a structured answer and supporting source IDs.
6. Validate those IDs and assemble citations from stored chunk metadata.

Uploads with identical bytes are deduplicated by SHA-256, even if renamed. Original
files are not retained: the extracted chunks and metadata are persisted in Chroma.
Chroma internally manages its own storage; there is no application SQLite database.

## Configuration

See `.env.example`. Defaults: `gpt-4.1-mini`, `text-embedding-3-small`, four retrieved
chunks, 1000-character chunks, 150-character overlap, and a 10 MiB upload limit.
Use an account-accessible chat model that supports structured output. API calls
have a 30-second request timeout and two retries. Embedding and chat calls are paid.

Keep the embedding model unchanged for an existing index. To change it, stop the
app, choose a new empty `CHROMA_DIR`, and re-upload documents. Chat model changes
do not require re-indexing.

## Docker

```bash
docker build -t document-qa .
docker run --rm -p 127.0.0.1:8000:8000 --env-file .env \
  -e CHROMA_DIR=/data/chroma -v document-qa-data:/data document-qa
```

The named volume retains the index when the container is replaced. Sessions are
in memory and reset on restart. Run one worker because histories are not shared
between processes. No separate Chroma container is needed.

## Tests

```bash
python -m pytest -q
```

Tests use a real temporary embedded Chroma index, generated text PDFs, a mocked
LLM, and deterministic test embeddings. They do not make paid API calls or download
ML models. They cover uploads, duplicates, page/snippet citations, refusal,
session isolation, persistence, input limits, and provider errors. These are
functional tests, not evidence of retrieval quality or real-model faithfulness.

## Prototype limits

One trusted owner and a small document collection. No authentication or user
isolation; session IDs separate histories, not access to documents. No OCR,
streaming, deletion, or background processing. A single lock serializes uploads
and chat requests. History keeps the latest ten turns for up to 100 sessions.

Scanned PDFs without readable text are rejected. Complex PDF layouts may extract
poorly. Citations are genuine source text, but semantic correctness and the model's
decision that an answer exists remain model-dependent. Document instructions are
treated as untrusted data in the prompt; this is not a guarantee against injection.
The upload limit bounds file bytes, not decompressed PDF memory use. Documents and
questions are sent to OpenAI as needed for embeddings and answers. Keep `.env` and
`data/` out of version control.
