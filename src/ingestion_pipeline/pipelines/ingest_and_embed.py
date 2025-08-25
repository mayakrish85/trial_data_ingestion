# src/ingestion_pipeline/pipelines/ingest_and_embed.py
from __future__ import annotations
import os, json, time, hashlib
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from tqdm.auto import tqdm

from ingestion_pipeline.preprocessing.doi_utils import normalize_doi
from ingestion_pipeline.chunking.chunkers import CHUNKERS, Chunk
from ingestion_pipeline.embeddings.embedders import resolve_embedder, EmbedConfig
from ingestion_pipeline.vectorstores.chroma_store import ChromaStore, ChromaConfig

def _load_fulltext(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    return data

def _doc_id_from_record(rec: Dict[str, Any]) -> str:
    doi = normalize_doi(rec.get("doi") or "")
    if doi:
        return doi
    # fallback: stable hash from title+journal
    base = f"{rec.get('title','')}|{rec.get('journal','')}"
    return "hash:" + hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]

@dataclass
class RunConfig:
    input_json: str
    persist_dir: str = "data/chroma"
    collection: str = "papers"
    reset_collection: bool = False

    chunker: str = "by_section"      # or "fixed"
    chunk_size: int = 1200
    chunk_overlap: int = 120

    embed_backend: str = "hf"        # "hf" or "openai"
    embed_model: str = "all-MiniLM-L6-v2"  # or "text-embedding-3-small"
    embed_batch: int = 64

    experiment: str = "exp1"         # label stored in metadata
    show_progress: bool = True

def run_ingest_and_embed(cfg: RunConfig) -> Dict[str, Any]:
    # Load
    records = _load_fulltext(cfg.input_json)

    # Chunker
    if cfg.chunker not in CHUNKERS:
        raise ValueError(f"Unknown chunker '{cfg.chunker}'. Options: {list(CHUNKERS.keys())}")
    chunker = CHUNKERS[cfg.chunker]

    # Embedder
    embedder = resolve_embedder(cfg.embed_backend, cfg.embed_model)
    e_cfg = EmbedConfig(model=cfg.embed_model, batch_size=cfg.embed_batch, show_progress=cfg.show_progress)

    # Chroma
    # Per-experiment collection name helps you compare setups; feel free to pass a fixed one instead.
    collection_name = cfg.collection or f"{cfg.experiment}:{cfg.embed_backend}:{cfg.embed_model}:{cfg.chunker}:{cfg.chunk_size}"
    store = ChromaStore(ChromaConfig(persist_dir=cfg.persist_dir, collection=collection_name,
                                     reset_collection=cfg.reset_collection))

    # Prepare all chunks
    doc_chunks: List[Tuple[str, Chunk, Dict[str, Any]]] = []
    it = records
    if cfg.show_progress:
        it = tqdm(records, desc="Preparing chunks", unit="doc")
    for rec in it:
        sections = rec.get("sections") or {}
        title = rec.get("title") or ""
        doc_id = _doc_id_from_record(rec)
        chunks = chunker.chunk(
            doc_id=doc_id, title=title, sections=sections,
            chunk_size=cfg.chunk_size, overlap=cfg.chunk_overlap
        )
        for ch in chunks:
            meta = {
                "doi": rec.get("doi"),
                "title": rec.get("title"),
                "journal": rec.get("journal"),
                "source": rec.get("source"),
                "pmcid": rec.get("pmcid"),
                "section_path": ch.section_path,
                "chunk_index": ch.chunk_index,
                "chunker": chunker.name,
                "chunk_size": cfg.chunk_size,
                "chunk_overlap": cfg.chunk_overlap,
                "embed_backend": cfg.embed_backend,
                "embed_model": cfg.embed_model,
                "experiment": cfg.experiment,
            }
            doc_chunks.append((doc_id, ch, meta))

    if not doc_chunks:
        return {"status": "no_chunks", "records": len(records)}

    # Deterministic IDs so reruns upsert the same chunks
    ids: List[str] = []
    texts: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    for doc_id, ch, meta in doc_chunks:
        ids.append(f"{doc_id}::c{ch.chunk_index}")
        texts.append(ch.text)
        metadatas.append(meta)

    # Embed
    embeds = embedder.embed(texts, cfg=e_cfg)

    # Upsert to Chroma
    store.upsert(ids=ids, embeddings=embeds, metadatas=metadatas, documents=texts,
                 show_progress=cfg.show_progress, batch_size=max(256, cfg.embed_batch))

    return {
        "status": "ok",
        "collection": store.collection.name,
        "persist_dir": cfg.persist_dir,
        "n_docs": len(records),
        "n_chunks": len(ids),
        "embed_backend": cfg.embed_backend,
        "embed_model": cfg.embed_model,
        "chunker": chunker.name,
        "chunk_size": cfg.chunk_size,
        "chunk_overlap": cfg.chunk_overlap,
        "experiment": cfg.experiment,
    }