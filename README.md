# Sherlock Holmes Mini-RAG

A command-line retrieval-augmented generation (RAG) project over five Sherlock Holmes stories. It loads the story files, splits them into sections and chunks, stores OpenAI embeddings in Qdrant, and uses retrieved excerpts to answer questions with source citations.

## How it works

1. Reads `.txt` files from `data/corpus/`.
2. Splits story text using heuristic section-header detection and paragraph/sentence boundaries. The configured target chunk size is 500 characters.
3. Embeds chunks with OpenAI's `text-embedding-3-small` model.
4. Stores vectors and chunk metadata in a Qdrant collection named `mp2_sherlock`.
5. Retrieves the three most relevant chunks for a question and asks `gpt-4o-mini` to produce an answer and citations.

## Project structure

```text
data/
  corpus/                    Sherlock Holmes story text files
  learner_questions.jsonl    Additional questions for validation
  predefined_questions.jsonl Built-in validation questions
src/
  models.py                  Model and chunking settings
  mp2_rag.py                 Ingestion, retrieval, answering, and CLI
requirements.txt             Python dependencies
```

## Requirements

- Python 3.10 or later
- An OpenAI API key
- A Qdrant endpoint (for example, a Qdrant Cloud cluster or a local Qdrant server)

## Setup

From the project root, install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Set the required environment variables in PowerShell. Replace the placeholders with your own credentials and endpoint; do not commit credentials to the repository.

```powershell
$env:OPENAI_API_KEY = "<your-openai-api-key>"
$env:QDRANT_URL = "<your-qdrant-url>"
$env:QDRANT_API_KEY = "<your-qdrant-api-key>"
```

For a local Qdrant server that does not require an API key, set `QDRANT_URL` to its URL and omit `QDRANT_API_KEY`.

## Usage

Run these commands from the project root:

```powershell
# Load, chunk, embed, and ingest the corpus
python src\mp2_rag.py ingest

# Ask questions interactively
python src\mp2_rag.py ask

# Run predefined and learner-question validation
python src\mp2_rag.py validate
```

Run ingestion before asking questions or validating. The application uses the existing `mp2_sherlock` collection if it already exists.

## Configuration

Defaults are defined in `src/models.py`:

| Setting | Default |
| --- | --- |
| Chat model | `gpt-4o-mini` |
| Embedding model | `text-embedding-3-small` |
| Embedding dimensions | `1536` |
| Qdrant collection | `mp2_sherlock` |
| Target chunk size | `500` characters |
| Temperature | `0.0` |

## Validation

The validation command checks answers against `data/predefined_questions.jsonl` and, when populated, `data/learner_questions.jsonl`. It reports whether the expected source was cited and how many expected answer facts matched.
