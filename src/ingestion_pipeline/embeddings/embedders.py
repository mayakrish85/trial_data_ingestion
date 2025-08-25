# src/ingestion_pipeline/embeddings/embedders.py
from __future__ import annotations
from typing import List, Iterable, Optional
from dataclasses import dataclass

from tqdm.auto import tqdm

# We keep both options; choose at runtime.
# HF (local) is default to avoid external deps/rate limits.

@dataclass
class EmbedConfig:
    model: str
    batch_size: int = 64
    show_progress: bool = True

class BaseEmbedder:
    name: str = "base"
    dim: Optional[int] = None
    def embed(self, texts: List[str], *, cfg: EmbedConfig) -> List[List[float]]:
        raise NotImplementedError

class HFEmbedder(BaseEmbedder):
    """SentenceTransformers (CPU/GPU)."""
    name = "hf"
    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self._model_name = model
        self._st = SentenceTransformer(model)
        try:
            self.dim = len(self._st.encode(["test"], show_progress_bar=False)[0])
        except Exception:
            self.dim = None

    def embed(self, texts: List[str], *, cfg: EmbedConfig) -> List[List[float]]:
        batch = max(1, int(cfg.batch_size))
        out: List[List[float]] = []
        it = range(0, len(texts), batch)
        if cfg.show_progress:
            it = tqdm(it, desc=f"Embeddings ({self._model_name})", unit="batch")
        for i in it:
            part = texts[i:i+batch]
            vecs = self._st.encode(part, show_progress_bar=False, convert_to_numpy=False)
            out.extend([v.tolist() if hasattr(v, "tolist") else list(v) for v in vecs])
        return out

class OpenAIEmbedder(BaseEmbedder):
    """OpenAI embeddings (requires OPENAI_API_KEY)."""
    name = "openai"
    def __init__(self, model: str = "text-embedding-3-small"):
        from openai import OpenAI
        self._client = OpenAI()
        self._model = model
        # infer dim from first call lazily

    def embed(self, texts: List[str], *, cfg: EmbedConfig) -> List[List[float]]:
        from openai import OpenAI
        client = self._client
        out: List[List[float]] = []
        batch = max(1, int(cfg.batch_size))
        it = range(0, len(texts), batch)
        if cfg.show_progress:
            it = tqdm(it, desc=f"Embeddings ({self._model})", unit="batch")
        for i in it:
            part = texts[i:i+batch]
            resp = client.embeddings.create(model=self._model, input=part)
            out.extend([d.embedding for d in resp.data])
        return out

def resolve_embedder(kind: str, model: str):
    kind = (kind or "hf").lower()
    if kind == "openai":
        return OpenAIEmbedder(model=model)
    return HFEmbedder(model=model)  # default
