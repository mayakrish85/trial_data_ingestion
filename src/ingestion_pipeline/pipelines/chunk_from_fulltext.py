from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from ingestion_pipeline.config.settings import get_settings
from ingestion_pipeline.utils.logger import get_logger
from ingestion_pipeline.chunking.chunker import TextChunker
from ingestion_pipeline.preprocessing.xml_cleaning import sections_to_text

logger = get_logger(__name__)

def chunk_from_fulltext(fulltext_json: str, output_dir: str, model_name: str | None = None) -> str:
    settings = get_settings()
    processed_dir = Path(output_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "chunks.parquet"
    model = model_name or settings.embedding_model
    chunker = TextChunker(model_name=model, max_tokens=settings.max_tokens, overlap=settings.chunk_overlap)
    rows = []
    with open(fulltext_json, "r", encoding="utf-8") as f:
        records = json.load(f)
    for rec in records:
        sections = rec.get("sections") or {}
        text = sections_to_text(sections)
        article_dict = {
            "id": rec.get("doi"),
            "title": rec.get("title"),
            "doi": rec.get("doi"),
            "url": None,
            "source": rec.get("source"),
            "full_text": text
        }
        rows.extend(chunker.chunk_article(article_dict))
    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    logger.info(f"Created {len(df)} chunks -> {out_path}")
    return str(out_path)
