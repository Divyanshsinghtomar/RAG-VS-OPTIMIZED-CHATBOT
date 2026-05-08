import faiss
import numpy as np


class FAISSVectorStore:

    def __init__(self, dimension: int):

        self.dimension = dimension

        self.index = faiss.IndexFlatIP(dimension)

        self.metadata = []

    def add_embeddings(self, embeddings: np.ndarray, metadata: list):

        embeddings = np.array(embeddings).astype("float32")

        self.index.add(embeddings)

        self.metadata.extend(metadata)

    def search(self, query_embedding: np.ndarray, top_k: int = 5):

        query_embedding = np.array(query_embedding).astype("float32")

        distances, indices = self.index.search(query_embedding, top_k)

        results = []

        for score, idx in zip(distances[0], indices[0]):

            results.append({"score": float(score), "metadata": self.metadata[idx]})

        return results
