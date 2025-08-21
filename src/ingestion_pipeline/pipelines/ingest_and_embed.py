from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from ingestion_pipeline.config.settings import get_settings
from ingestion_pipeline.utils.logger import get_logger
from ingestion_pipeline.preprocessing.normalize import ingest_to_jsonl
from ingestion_pipeline.chunking.chunker import TextChunker
from ingestion_pipeline.embeddings.embedder import Embedder

logger = get_logger(__name__)

def ingest_stage(input_path: str, output_dir: str) -> str:
    settings = get_settings()
    processed_dir = Path(output_dir) / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "articles.jsonl"
    count = ingest_to_jsonl(input_path, str(out_path))
    logger.info(f"Ingested {count} records -> {out_path}")
    return str(out_path)

def chunk_stage(articles_jsonl: str, output_dir: str, model_name: str | None = None) -> str:
    settings = get_settings()
    processed_dir = Path(output_dir) / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "chunks.parquet"
    model = model_name or settings.embedding_model
    chunker = TextChunker(model_name=model, max_tokens=settings.max_tokens, overlap=settings.chunk_overlap)
    rows = []
    with open(articles_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            rows.extend(chunker.chunk_article(rec))
    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    logger.info(f"Created {len(df)} chunks -> {out_path}")
    return str(out_path)

def embed_stage(chunks_parquet: str, output_dir: str, model_name: str | None = None) -> str:
    settings = get_settings()
    emb_dir = Path(output_dir) / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    out_path = emb_dir / "embeddings.parquet"
    model = model_name or settings.embedding_model
    df = pd.read_parquet(chunks_parquet)
    embedder = Embedder(model_name=model, batch_size=settings.batch_size)
    df_emb = embedder.embed_chunk_df(df)
    df_emb.to_parquet(out_path, index=False)
    logger.info(f"Embedded {len(df_emb)} chunks -> {out_path}")
    return str(out_path)

def run_all(input_path: str, output_dir: str, model_name: str | None = None) -> dict:
    articles = ingest_stage(input_path, output_dir)
    chunks = chunk_stage(articles, output_dir, model_name=model_name)
    embeddings = embed_stage(chunks, output_dir, model_name=model_name)
    return {"articles_jsonl": articles, "chunks_parquet": chunks, "embeddings_parquet": embeddings}
