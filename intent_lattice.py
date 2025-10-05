"""
Intent Lattice Module
---------------------
Handles sub-intent extraction and redundancy computation.
"""

import numpy as np

class IntentLattice:
    def __init__(self, m: int = 4, seed: int = 42):
        """
        m : number of sub-intents
        """
        np.random.seed(seed)
        self.m = m

    def encode_query(self, query: str) -> np.ndarray:
        """
        Dummy simulation of intent vector π for demonstration.
        In real use, plug a transformer encoder + clustering.
        """
        vec = np.random.dirichlet(np.ones(self.m))
        return vec

    def encode_candidates(self, n: int) -> np.ndarray:
        """
        Produce sub-intent distributions for each candidate doc.
        """
        return np.array([np.random.dirichlet(np.ones(self.m)) for _ in range(n)])

    def compute_redundancy(self, embeddings: np.ndarray,
                           intents: np.ndarray,
                           alpha: float = 0.6) -> np.ndarray:
        """
        Compute hybrid redundancy matrix.
        """
        n = len(embeddings)
        sim = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                cos = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-8
                )
                intent_sim = np.dot(intents[i], intents[j])
                sim[i, j] = alpha * cos + (1 - alpha) * intent_sim
                sim[j, i] = sim[i, j]
        return sim
