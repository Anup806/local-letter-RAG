from typing import List, Dict, Any

import chromadb

from app.config import CHROMA_DIR


def get_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=CHROMA_DIR)


def get_collection(client: chromadb.PersistentClient, name: str) -> chromadb.Collection:
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


def _embed_texts(embedder, texts: List[str]) -> List[List[float]]:
    emb = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [e.tolist() for e in emb]


def add_chunks(collection, doc_id: str, chunks: List[Dict[str, Any]], embedder) -> int:
    ids = [f"{doc_id}_{c['chunk_index']}" for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [
        {"doc_id": doc_id, "page": c["page"], "chunk_index": c["chunk_index"]}
        for c in chunks
    ]
    embeddings = _embed_texts(embedder, documents)

    collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    return len(ids)


def query_chunks(collection, query: str, embedder, top_k: int) -> List[Dict[str, Any]]:
    query_embedding = _embed_texts(embedder, [query])[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    ids = results.get("ids", [[]])[0]

    items = []
    for doc, meta, dist, item_id in zip(docs, metas, distances, ids):
        items.append({
            "id": item_id,
            "text": doc,
            "metadata": meta or {},
            "distance": dist,
        })
    return items
