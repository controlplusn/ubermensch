from __future__ import annotations

import chromadb
 
from dataclasses import dataclass
from pathlib import Path
 
from vault.cli.logger import done, log, step, warn
 
CHROMA_PATH = Path.home() / ".vault" / "store" / "chroma"
COLLECTION_NAME = "vault_notes"

@dataclass
class RetrievedChunk:
    note_title: str
    note_path:  str
    text: str
    score: float
    chunk_id: str


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

def upsert_chunks(chunks, embeddings: list[list[float]]) -> None:
    if not chunks:
        warn("STORE", "No chunks to upsert — skipping")
        return
    
    step("STORE", "Upserting chunks into ChromaDB vector store")
    log(f"Store location: {CHROMA_PATH}")
    log(f"Collection: '{COLLECTION_NAME}'  |  Metric: cosine similarity")
    log("Upsert is idempotent — unchanged chunks are skipped automatically")
    log(f"Writing {len(chunks)} chunk(s)...")
 
    collection = get_collection()

    ids = [c.chunk_id for c in chunks]
    documents = [c.text for c in chunks]
    metadatas = [
        {"note_title": c.note_title, "note_path": str(c.note_path)}
        for c in chunks
    ]

    # Batch upsert in groups of 500 (ChromaDB recommended batch size)
    BATCH = 500
    for i in range(0, len(ids), BATCH):
        batch_ids   = ids[i:i + BATCH]
        batch_docs  = documents[i:i + BATCH]
        batch_meta  = metadatas[i:i + BATCH]
        batch_emb   = embeddings[i:i + BATCH]
 
        collection.upsert(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_meta,
            embeddings=batch_emb,
        )
        log(f"  Batch {i // BATCH + 1}: upserted {len(batch_ids)} chunks")
 
    total = collection.count()
    done("STORE", f"Vector store now contains {total} total chunk(s)")

def retrieve(
    query_embedding: list[float],
    top_k: int = 5,
    tag_filter: str | None = None,
) -> list[RetrievedChunk]:
    # tag_filter: optional note title substring filter (future use)

    step("RETRIEVE", f"Searching vector store for top {top_k} relevant chunks")
    log("Computing cosine similarity between query and all stored embeddings")
    log("HNSW index makes this fast even across thousands of chunks")

    collection = get_collection()
    total = collection.count()

    if total == 0:
        warn("RETRIEVE", "Vector store is empty — run `vault init` first")
        return []
 
    log(f"Searching across {total} indexed chunks")
 
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, total),
        include=["documents", "metadatas", "distances"],
    )
 
    chunks: list[RetrievedChunk] = []
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]
    ids = results["ids"][0]

    for doc, meta, dist, cid in zip(docs, metas, distances, ids):
        # ChromaDB cosine distance ∈ [0, 2]; convert to similarity ∈ [0, 1]
        score = round(1 - dist / 2, 4)
        chunks.append(RetrievedChunk(
            note_title=meta.get("note_title", "Unknown"),
            note_path=meta.get("note_path", ""),
            text=doc,
            score=score,
            chunk_id=cid,
        ))
        log(f"  [{score:.3f}]  {meta.get('note_title', '?')}")
 
    done("RETRIEVE", f"Retrieved {len(chunks)} chunk(s)")
    return chunks

def store_stats() -> dict:
    try:
        collection = get_collection()
        return {"total_chunks": collection.count()}
    except Exception:
        return {"total_chunks": 0}