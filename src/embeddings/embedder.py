from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingModel:

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):

        self.model = SentenceTransformer(model_name)

    def encode(self, texts, normalize: bool = True) -> np.ndarray:

        embeddings = self.model.encode(
            texts, normalize_embeddings=normalize, batch_size=64, show_progress_bar=True
        )

        return embeddings
