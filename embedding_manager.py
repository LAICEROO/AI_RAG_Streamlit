from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List

class EmbeddingManager:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.dimension = 384  # Dimension of all-MiniLM-L6-v2 embeddings
        self.index = faiss.IndexFlatL2(self.dimension)
        self.texts = []

    def create_embeddings(self, texts: List[str]):
        embeddings = self.model.encode(texts)
        self.index.add(np.array(embeddings).astype('float32'))
        self.texts.extend(texts)

    def search(self, query: str, k: int = 3) -> List[str]:
        query_embedding = self.model.encode([query])
        D, I = self.index.search(np.array(query_embedding).astype('float32'), k)
        return [self.texts[i] for i in I[0]]

    def clear(self):
        self.index = faiss.IndexFlatL2(self.dimension)
        self.texts = []