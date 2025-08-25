# src/ingestion_pipeline/chunking/chunkers.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Iterable, List, Tuple, Optional

import re

# --------- utilities ---------

def _iter_section_texts(sections: Dict[str, Any], path: Optional[List[str]] = None):
    """
    Very forgiving traversal of your nested sections dict.
    Yields (section_path, text) for any subtree that has text.
    """
    path = path or []
    if sections is None:
        return
    if isinstance(sections, dict):
        # Common patterns: {"Some Heading": {...}}, each child may have "text" and nested keys.
        for key, val in sections.items():
            next_path = path + ([str(key)] if key is not None else [])
            if isinstance(val, dict):
                # Text directly on node?
                txt = val.get("text")
                if isinstance(txt, str) and txt.strip():
                    yield (" / ".join(next_path), txt)
                # Recurse children
                for child_k, child_v in val.items():
                    if isinstance(child_v, (dict, list)):
                        yield from _iter_section_texts({child_k: child_v}, next_path)
            elif isinstance(val, str) and val.strip():
                # Some structures store just strings
                yield (" / ".join(next_path), val)
            elif isinstance(val, list):
                for i, item in enumerate(val):
                    if isinstance(item, str) and item.strip():
                        yield (" / ".join(next_path + [str(i)]), item)
                    elif isinstance(item, dict):
                        yield from _iter_section_texts(item, next_path + [str(i)])
    elif isinstance(sections, list):
        for i, item in enumerate(sections):
            if isinstance(item, str) and item.strip():
                yield (" / ".join((path or []) + [str(i)]), item)
            elif isinstance(item, dict):
                yield from _iter_section_texts(item, (path or []) + [str(i)])


def _split_text_windows(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Word-boundary splitter with overlap (by characters, aligned to whitespace).
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    if chunk_size <= 0:
        return [text]

    chunks = []
    start = 0
    n = len(text)
    # ensure reasonable overlap
    overlap = max(0, min(overlap, max(0, chunk_size - 1)))
    step = max(1, chunk_size - overlap)

    while start < n:
        end = min(n, start + chunk_size)
        # expand end to next whitespace to avoid cutting words (but don't exceed +40)
        if end < n:
            bump = text[end:end+40]
            m = re.search(r"\s", bump)
            if m:
                end = end + m.start()
        chunks.append(text[start:end].strip())
        if end == n:
            break
        start = end - overlap
    return [c for c in chunks if c]

# --------- chunkers ---------

@dataclass
class Chunk:
    doc_id: str
    chunk_index: int
    text: str
    section_path: Optional[str] = None

class BaseChunker:
    name: str = "base"
    def chunk(self, *, doc_id: str, title: str, sections: Dict[str, Any],
              chunk_size: int, overlap: int) -> List[Chunk]:
        raise NotImplementedError

class BySectionChunker(BaseChunker):
    """
    One chunk per section (long sections are further windowed).
    """
    name = "by_section"
    def chunk(self, *, doc_id: str, title: str, sections: Dict[str, Any],
              chunk_size: int = 1200, overlap: int = 120) -> List[Chunk]:
        out: List[Chunk] = []
        idx = 0
        # If no obvious structure, fall back to entire article
        collected = list(_iter_section_texts(sections))
        if not collected:
            collected = [(None, _coalesce_all_text(sections))]
        for path, txt in collected:
            if not txt or not txt.strip():
                continue
            parts = _split_text_windows(txt, chunk_size, overlap) if chunk_size else [txt]
            for p in parts:
                out.append(Chunk(doc_id=doc_id, chunk_index=idx, text=p, section_path=path))
                idx += 1
        return out

class FixedWindowChunker(BaseChunker):
    """
    Flatten whole article to one string, then fixed windows.
    """
    name = "fixed"
    def chunk(self, *, doc_id: str, title: str, sections: Dict[str, Any],
              chunk_size: int = 1200, overlap: int = 120) -> List[Chunk]:
        flat = _coalesce_all_text(sections)
        parts = _split_text_windows(flat, chunk_size, overlap) if chunk_size else [flat]
        return [Chunk(doc_id=doc_id, chunk_index=i, text=p) for i, p in enumerate(parts)]

def _coalesce_all_text(sections: Dict[str, Any]) -> str:
    texts = []
    for _, txt in _iter_section_texts(sections):
        texts.append(txt)
    return "\n\n".join(texts).strip()

# Registry
CHUNKERS = {
    BySectionChunker.name: BySectionChunker(),
    FixedWindowChunker.name: FixedWindowChunker(),
}
