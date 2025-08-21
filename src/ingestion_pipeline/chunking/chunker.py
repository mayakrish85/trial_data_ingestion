from __future__ import annotations
from typing import List, Dict, Optional
from transformers import AutoTokenizer

def _count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))

def _split_by_tokens(tokenizer, text: str, max_tokens: int, overlap: int) -> List[str]:
    if not text or text.strip() == "":
        return []
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text.strip())
        if end == len(tokens):
            break
        start = max(0, end - overlap)
    return chunks

class TextChunker:
    def __init__(self, model_name: str, max_tokens: int = 512, overlap: int = 50):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.max_tokens = max_tokens
        self.overlap = overlap

    def chunk_article(self, article: Dict, text_field: str = "full_text", article_id_field: str = "id") -> List[Dict]:
        text: Optional[str] = article.get(text_field)
        if not text:
            return []
        base_id = article.get(article_id_field) or ""
        chunks = _split_by_tokens(self.tokenizer, text, self.max_tokens, self.overlap)
        out = []
        for i, ch in enumerate(chunks):
            out.append({
                "article_id": base_id,
                "chunk_id": f"{base_id}::chunk_{i}",
                "text": ch,
                "n_tokens": _count_tokens(self.tokenizer, ch),
                "meta": {
                    "title": article.get("title"),
                    "doi": article.get("doi"),
                    "url": article.get("url"),
                    "source": article.get("source"),
                },
            })
        return out
