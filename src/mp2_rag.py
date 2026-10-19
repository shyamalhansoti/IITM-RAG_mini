"""MP2 · Mini-RAG — Starter Template
====================================

You'll build a complete RAG pipeline over the Sherlock Holmes corpus in this
file. Fill in every TODO. The reference solution is ~250 lines, but yours can
be shorter or longer — what matters is that it works end-to-end.

Pipeline you're building:
    corpus/*.txt  →  chunks  →  embeddings  →  Qdrant
                                                  ↓
                              question  →  retrieve  →  answer + citations

Run sequence (once you've filled in the TODOs):
    pip install -r requirements.txt
    source .env                 # exports your OpenAI + Qdrant credentials
    python mp2_rag.py ingest    # builds the collection (run once)
    python mp2_rag.py ask       # interactive Q&A loop
    python mp2_rag.py validate  # runs against data/predefined_questions.jsonl

Tip: get the CORE pipeline working FIRST (Steps 1-7 below), THEN come back to
polish and add your 3 questions. Don't try to perfect each step before moving
on — you'll learn more from a rough end-to-end loop than a polished half.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from models import Settings
import numpy as np

# ─── Configuration ──────────────────────────────────────────────────────

CORPUS_DIR        = Path(__file__).parent.parent / "data/corpus"
DATA_DIR          = Path(__file__).parent.parent / "data"
chunk_counter=0
BASE_DIR = Path(__file__).resolve().parent.parent
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

client = OpenAI()

qdrant = QdrantClient(
    url=os.environ.get("QDRANT_URL"),
    api_key=os.environ.get("QDRANT_API_KEY"),
)

settings=Settings()

ANSWER_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "answer_question",
        "description": (
            "Return a structured answer with content and sources."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The answer in 2-4 sentences.",
                },
                "citations": {
                    "type": "array",
                    "items": { "type": "object",
                                "properties": {
                        
                        "source":{"type":"string", "description": "source is the txt file"},"section":{"type":"string", "description": "section"},
                    
                    },
                    },
                    "description": (
                        "Source identifiers or URLs you used. Empty list is fine "
                        "if you used general knowledge."
                    ),
                },
            },
            "required": ["source","section","content", "citations"],
        },
    },
}


# ─── Step 1: Load the corpus ────────────────────────────────────────────

def load_corpus(corpus_dir: Path) -> list[dict[str, Any]]:
    """Read every .txt file in the corpus directory.

    Returns a list of dicts, each with: source (filename), title (first line),
    and text (full content).

    TODO:
      - Iterate over every *.txt file in corpus_dir (use Path.glob)
      - For each file, read its text and extract the first non-empty line as title
      - Return the list of doc dicts
    """
    # TODO: your code here
    corpus_list:list[dict[str,Any]]=[]

    if(os.path.isdir(corpus_dir)):
        for file_path in corpus_dir.iterdir():
            # print(files)
            with open(file_path,"r",encoding="utf-8") as file:
                # print(file.readline().strip())
                new_item: dict[str,Any]={"source":file_path.name,"title":file.readline().strip(),"text":file.read()}
                corpus_list.append(new_item)
            # break

    # print("=====corpus===========")
    # print(corpus_list)
    return corpus_list
    # raise NotImplementedError("Implement load_corpus")


# ─── Step 2: Chunk each document ────────────────────────────────────────
def is_section_header(line: str) -> bool:
        """
        Heuristic: Detects short lines without terminal punctuation.
        Also filters out numeric-only lines (like page numbers).
        """
        clean_line = line.strip()
        
        # Empty lines are not headers
        if not clean_line:
            return False
            
        # Ignore lines that are just numbers (e.g., page numbers)
        if clean_line.isdigit():
            return False
            
        # Check lengths (headers are typically short, under 80 chars / ~12 words)
        if len(clean_line) > 80 or len(clean_line.split()) > 12:
            return False
            
        # Check for missing terminal punctuation (. ! ?)
        if clean_line[-1] in ['.', '!', '?']:
            return False
            
        return True


def parse_sections(doc: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Splits text into sections based on heuristic header detection.
        """
        text_content = doc["text"]
        lines = text_content.splitlines()
    
        # print(item["title"],lines)
        sections = []   
        current_header =doc["title"]
        current_content = []

        for line in lines:
            if is_section_header(line):
                # Save the previous section if it has content
                if current_content:
                    sections.append({
                        "source":doc["source"],
                        "title":doc["title"],
                        "header": current_header,
                        "text": "\n".join(current_content).strip()
                    })
                current_header = line.strip()
                current_content = []
            else:
                current_content.append(line)

        # Append final section
        if current_content:
            sections.append({
                "header": current_header,
                "text": "\n".join(current_content).strip()
            })
    
        return sections


def chunk_text(item: dict[str,Any]) -> list[dict[str, Any]]:

        """
        Chunks the text while preserving section context and structural boundaries.
        """
        sections = parse_sections(item)
        chunks = []
        # global chunk_counter
        for section in sections:
            header = section["header"]
            content = section["text"]
            
            # Split content by paragraphs first to respect text structure
            paragraphs = content.split("\n\n")
            
            current_chunk = []
            current_size = 0
            
            for para in paragraphs:
                para_size = len(para)
                
                # If a single paragraph is larger than max size, split by sentences
                if para_size > settings.TARGET_CHUNK_SIZE:
                    sentences = re.split(r'(?<=[.!?])\s+', para)
                    for sentence in sentences:
                        if current_size + len(sentence) > settings.TARGET_CHUNK_SIZE:
                            if current_chunk:
                                chunks.append({
                                    "source":item["source"],
                                    "title":item["title"],
                                    "header": header,
                                    "text": " ".join(current_chunk)
                                })
                                # chunk_counter+=1
                            # Start new chunk with overlap if applicable
                            current_chunk = current_chunk[-1:] if current_chunk else []
                            current_size = sum(len(s) for s in current_chunk)
                        
                        current_chunk.append(sentence)
                        current_size += len(sentence)
                        
                else:
                    # Normal paragraph accumulation
                    if current_size + para_size > settings.TARGET_CHUNK_SIZE:
                        if current_chunk:
                            chunks.append({
                                
                                "source":item["source"],
                                "title":item["title"],
                                "header": header,
                                "text": "\n\n".join(current_chunk)
                            })
                            # chunk_counter+=1
                        current_chunk = []
                        current_size = 0
                    
                    current_chunk.append(para)
                    current_size += para_size
                    

            # Append the remaining text in the section
            if current_chunk:
                chunks.append({
                    "source":item["source"],
                    "title":item["title"],
                    "header": header,
                    "text": "\n\n".join(current_chunk) if "\n\n" in content else " ".join(current_chunk)
                })
                # chunk_counter+=1

        return chunks


def chunk_document(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Split a document into smaller chunks.

    Each chunk should be a dict with: source, title, section, text.

    Approach (your choice):
      - Simple: fixed-size windows (split text into N-character chunks with overlap)
      - Smarter: split on paragraph boundaries (\\n\\n), then pack paragraphs
        into chunks up to TARGET_CHUNK_SIZE characters

    The reference solution uses the smarter approach, plus heuristic
    section-header detection (short lines without terminal punctuation).
    Either approach is acceptable.

    TODO:
      - Pick an approach
      - Implement it
      - Return list of chunk dicts
    """
    # TODO: your code here
    chunks : list[dict[str, Any]] = []
    # for item in doc: 
        # chunks_item=chunk_text(item)
        # chunks.append(chunk_text(doc["text"]))
        #print(chunks_item)
        # break
        # print(item["title"])
    chunks=chunk_text(doc)
    return chunks
    # raise NotImplementedError("Implement chunk_document")


# ─── Step 3: Embed text ─────────────────────────────────────────────────

def embed_texts(texts_to_embed: list[str]) -> list[list[float]]:
    """Batch-embed a list of texts using OpenAI's embedding model.

    Returns a list of 1536-dim float vectors (same order as inputs).

    TODO:
      - Call openai.embeddings.create with EMBEDDING_MODEL and the texts
      - Extract the embedding vectors from the response
    """
    # TODO: your code here
    # resp = openai().embeddings.create(model=EMBED_MODEL, input=texts)
    response = client.embeddings.create(
        input=texts_to_embed,
        model=settings.EMBEDDING_MODEL
    )
    return response
    # raise NotImplementedError("Implement embed_texts")


# ─── Step 4: Set up the Qdrant collection ───────────────────────────────

def setup_collection() -> None:
    """Create (or recreate) the Qdrant collection.

    TODO:
      - Use qdrant.recreate_collection
      - VectorParams with EMBEDDING_DIM and Distance.COSINE
    """
    # TODO: your code here
    if not qdrant.collection_exists(collection_name=settings.COLLECTION_NAME):
        qdrant.create_collection(
            collection_name=settings.COLLECTION_NAME,
            vectors_config=VectorParams(
                size=settings.EMBEDDING_DIM, 
                distance=Distance.COSINE 
            )
        )
        print(f"Collection '{settings.COLLECTION_NAME}' created successfully.")
    else:
        print(f"Collection '{settings.COLLECTION_NAME}' already exists.")

# ─── Step 5: Ingest chunks into Qdrant ──────────────────────────────────

def ingest_chunks(chunks: list[dict[str, Any]]) -> None:
    """Embed every chunk and upsert into Qdrant.

    TODO:
      - Call embed_texts on the chunk texts
      - Build PointStruct objects (id=uuid, vector, payload=chunk dict)
      - qdrant.upsert
    """
    # setup_collection()

    # text_to_embed = [d["text"] for d in chunks[0]]
    text_to_embed: list[str]=[]
    # for dictionary in chunks:
    for item  in chunks:
        text_to_embed.append(item["text"])
        # print("======text_embed=================",text_to_embed)
    
    
    embedded_texts =embed_texts(text_to_embed)
    embeddings = [data.embedding for data in embedded_texts.data]

    # TODO: your code here
     # Map into Qdrant Point structures
    points = []
    # for dictionary in chunks:
    for idx, chunk in enumerate(chunks):
            points.append(
                PointStruct(
                    id=uuid.uuid4(),  # Using the deterministic string ID generated by the chunker
                    vector=embeddings[idx],
                    payload={
                        "chunk_id":f"{idx}_{chunk['source']}",
                        "source":chunk["source"],
                        "title":chunk["title"],
                        "header": chunk["header"],
                        "text": chunk["text"],
                    
                    }
                )
            )
            
    # Batch upsert points to Qdrant
    operation_info = qdrant.upsert(
        collection_name=settings.COLLECTION_NAME,
        wait=True,
        points=points
    )
    print(f"Successfully uploaded chunks to Qdrant. Status: {operation_info.status}")  
    # raise NotImplementedError("Implement ingest_chunks")


# ─── Step 6: Retrieve ───────────────────────────────────────────────────

def retrieve(query: str, k: int = 3) -> list[dict[str, Any]]:
    """Retrieve top-k chunks for a query.

    TODO:
      - Embed the query
      - qdrant.search with the query vector, limit=k
      - Return list of chunk dicts (include score for citations)
    """
    # TODO: your code here
    q_vec = client.embeddings.create(input=query,model=settings.EMBEDDING_MODEL)
    
    # print(store.collection_size)
    results = qdrant.query_points(
        collection_name=settings.COLLECTION_NAME,
        query=q_vec.data[0].embedding,
        limit=k,
        with_vectors=True,  # <-- CRITICAL: Tells Qdrant to return the raw float arrays
        with_payload=True
    ).points
   
    return [
        {
            "chunk_id":  h.payload["chunk_id"],
            "title":h.payload["title"],
            "section":h.payload["header"],
            "source": h.payload["source"],
            "text":      h.payload["text"],
            "score":     h.score,
            
        }
        for h in results
    ]

    # raise NotImplementedError("Implement retrieve")


# ─── Step 7: Generate the answer ────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful assistant answering questions about a small
collection of Sherlock Holmes stories. You will be given the user's question and
several relevant excerpts. Use ONLY the provided excerpts to answer. If the
excerpts don't contain the answer, say so plainly. Cite the source (story title
+ section) in your answer."""

def format_retrieved_chunks(chunks):
    formatted_chunks = []

    for i, chunk in enumerate(chunks, start=1):
        formatted = (
            # f"[source {i}]\n"
            f"source: {chunk['source']}\n"
            f"title: {chunk['title']}\n"
            f"section: {chunk['section']}\n"
            f"chunk_id: {chunk['chunk_id']}\n"
            f"Content: {chunk['text']}"
        )

        formatted_chunks.append(formatted)

    return "\n\n".join(formatted_chunks)


def answer(question: str, k: int = 3) -> dict[str, Any]:
    """End-to-end: retrieve, format context, call LLM, return result.

    TODO:
      - Call retrieve(question, k=k)
      - Format the retrieved chunks into a context string
        (include "[Source: <title> — <section>]" before each)
      - Call openai.chat.completions.create with SYSTEM_PROMPT and the user message
      - Return dict with: question, answer, citations, latency_ms
    """
    # TODO: your code here
    # query="who was Victor Hatherley?"
    try:

        retrieval_data=retrieve(question,3)
        context = format_retrieved_chunks(retrieval_data)
        # print(context)  
        started=time.time()
        response = client.chat.completions.create(
            model=settings.CHAT_MODEL,
            temperature=0.0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content":
                    f"Context:\n{context}\n\n---\n\nQuestion: {question}"},
            ],
            tools=[ANSWER_TOOL],
                tool_choice={
                    "type": "function",
                    "function": {"name": "answer_question"},
                },
        )

        tool_calls = response.choices[0].message.tool_calls
        args_json = tool_calls[0].function.arguments
        args = json.loads(args_json)

        # print("===========args",args)
        latency = time.time() - started
        result={
                "question": question,
                # "source":args["source"],
                # "title":args["title"],
                "answer":   args["content"],  
                "citations": args["citations"], 
                # "section":  args["section"],            
                "latency_ms":  latency,
        }
        # print("===========REsult==========",result)
        return result 
    
    except Exception as e:
        print(f"Function failed at error: {e}")
        return {"error": str(e), "question": question}

# ─── Validation harness (provided — do not modify) ──────────────────────
def validate_against(jsonl_path: Path) -> None:
    questions = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
    print(f"\n  Validating {len(questions)} questions from {jsonl_path.name}…\n")

    hits = 0
    for q in questions:
        result = answer(q["question"], k=3)
        # print(result)
        cited_sources = {cit["source"] for cit in result["citations"]}
        source_hit = q["expected_source"] in cited_sources

        ans_lower = result["answer"].lower()
        facts_hit = sum(1 for fact in q.get("expected_facts", []) if fact.lower() in ans_lower)
        facts_total = len(q.get("expected_facts", []))

        verdict = "✓" if source_hit else "✗"
        print(f"  {verdict} {q['id']}")
        print(f"      Q: {q['question']}")
        print(f"      Cited: {', '.join(cited_sources)}")
        print(f"      A: {ans_lower}")
        print(f"      Expected: {q['expected_source']}")
        print(f"      Facts matched: {facts_hit}/{facts_total}")
        print(f"      Latency: {result.get('latency_ms', '?')}ms")
        print()
        if source_hit:
            hits += 1

    print(f"  Source-match: {hits}/{len(questions)}")


# ─── CLI (provided — do not modify) ─────────────────────────────────────
def cmd_ingest() -> None:
    print("→ Loading corpus…")
    docs = load_corpus(CORPUS_DIR)
    print(f"  {len(docs)} documents loaded")

    print("→ Chunking…")
    all_chunks: list[dict[str, Any]] = []
    for doc in docs:
        # print("==============doc===================",doc)
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
        print(f"  {doc['source']}: {len(chunks)} chunks")
        # break

    # print(all_chunks)
       
    print(f"→ Total chunks: {len(all_chunks)}")
    print("→ Setting up Qdrant collection…")
    setup_collection()

    print("→ Ingesting…")
    ingest_chunks(all_chunks)
    print("\n✓ Done. Try: python mp2_rag.py ask")


def cmd_ask() -> None:
    print("Mini-RAG over the Sherlock Holmes corpus.")
    print("Type your question. Empty line or Ctrl-C to exit.\n")
    while True:
        try:
            q = input("? ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not q:
            return
        result = answer(q, k=3)
        # print(result)
        print(f"{result['answer']}\n")
        print("  Sources:")
        for c in result["citations"]:
            print(f"    - {c['source']} — {c['section']}")
        print(f"  Latency: {result.get('latency_ms', '?')}ms\n")


def cmd_validate() -> None:
    validate_against(DATA_DIR / "predefined_questions.jsonl")
    learner_path = DATA_DIR / "learner_questions.jsonl"
    if learner_path.exists():
        first = json.loads(learner_path.read_text().splitlines()[0])
        if not first["question"].startswith("Replace this"):
            validate_against(learner_path)


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)
    cmd = sys.argv[1]
    if cmd == "ingest":   cmd_ingest()
    elif cmd == "ask":    cmd_ask()
    elif cmd == "validate": cmd_validate()
    else:
        print(f"Unknown command: {cmd}\n")
        print(__doc__)
        sys.exit(1)




if __name__ == "__main__":

    main()
    # corpus=load_corpus(CORPUS_DIR)
    # chunks=chunk_document(corpus)
    # ingest_chunks(chunks)
       
    # Delete the collection
    # qdrant.delete_collection(collection_name=settings.COLLECTION_NAME)
    