from __future__ import annotations
from typing import Iterable
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name: str = "intfloat/e5-base-v2", batch_size: int = 32):
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size

    def encode_texts(self, texts: Iterable[str]) -> np.ndarray:
        return self.model.encode(
            [f"passage: {t}" for t in texts],
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    def embed_chunk_df(self, chunks_df: pd.DataFrame) -> pd.DataFrame:
        embeddings = self.encode_texts(chunks_df["text"].tolist())
        chunks_df = chunks_df.copy()
        chunks_df["embedding"] = embeddings.tolist()
        chunks_df["embedding_dim"] = embeddings.shape[1]
        return chunks_df
