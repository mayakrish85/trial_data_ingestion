# src/ingestion_pipeline/vectorstores/chroma_store.py
from __future__ import annotations
from typing import List, Dict, Any, Iterable, Optional
from dataclasses import dataclass
from tqdm.auto import tqdm
import chromadb
from chromadb.config import Settings

@dataclass
class ChromaConfig:
    persist_dir: str = "data/chroma"
    collection: str = "papers"
    distance: str = "cosine"  # 'cosine' | 'l2' | 'ip'
    reset_collection: bool = False

class ChromaStore:
    def __init__(self, cfg: ChromaConfig):
        self._cfg = cfg
        self._client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=cfg.persist_dir
        ))
        if cfg.reset_collection:
            try:
                self._client.delete_collection(cfg.collection)
            except Exception:
                pass
        self._col = self._client.get_or_create_collection(
            name=cfg.collection,
            metadata={"hnsw:space": cfg.distance}
        )

    @property
    def collection(self):
        return self._col

    def upsert(self, *, ids: List[str], embeddings: List[List[float]],
               metadatas: List[Dict[str, Any]], documents: List[str],
               show_progress: bool = True, batch_size: int = 512):
        n = len(ids)
        it = range(0, n, batch_size)
        if show_progress:
            it = tqdm(it, total=(n + batch_size - 1) // batch_size, desc="Chroma upsert", unit="batch")
        for i in it:
            j = min(n, i + batch_size)
            self._col.upsert(
                ids=ids[i:j],
                embeddings=embeddings[i:j],
                metadatas=metadatas[i:j],
                documents=documents[i:j],
            )
        # Persist to disk
        try:
            self._client.persist()
        except Exception:
            pass
