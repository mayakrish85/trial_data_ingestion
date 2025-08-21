from __future__ import annotations
from typing import Iterable, List, Dict, Any
import json, pandas as pd
from pathlib import Path
from .bib_parser import parse_bib_file
from ingestion_pipeline.data_models.article import Article

SUPPORTED_EXT = {".csv", ".json", ".jsonl", ".ndjson", ".bib"}

def _load_records(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    ext = p.suffix.lower()
    if ext not in SUPPORTED_EXT:
        raise ValueError(f"Unsupported input format: {ext}")
    if ext == ".csv":
        df = pd.read_csv(p)
        return df.to_dict(orient="records")
    if ext == ".json":
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return [data]
        return data
    if ext in {".jsonl", ".ndjson"}:
        out = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out
    if ext == ".bib":
        return parse_bib_file(path)
    raise RuntimeError("Unreachable")

def normalize_records(records: Iterable[Dict[str, Any]]) -> List[Article]:
    articles: List[Article] = []
    for r in records:
        art = Article(
            id=r.get("id") or r.get("pmcid") or r.get("pmid"),
            title=r.get("title"),
            doi=r.get("doi"),
            url=r.get("url"),
            source=r.get("source"),
            full_text=r.get("full_text") or r.get("text") or r.get("body"),
            meta=r,
        )
        articles.append(art)
    return articles

def ingest_to_jsonl(input_path: str, output_jsonl_path: str) -> int:
    records = _load_records(input_path)
    articles = normalize_records(records)
    count = 0
    with open(output_jsonl_path, "w", encoding="utf-8") as f:
        for art in articles:
            f.write(art.model_dump_json() + "\n")
            count += 1
    return count
