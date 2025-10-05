"""
Demo Experiment for Q-HIVE
--------------------------
Generates dummy data and runs end-to-end pipeline.
"""

import numpy as np
from quantum.config import QHIVE_PARAMS, SEED
from quantum.intent_lattice import IntentLattice
from quantum.qmmr_optimizer import QMMROptimizer
from quantum.evaluation import ndcg_at_k, intent_coverage, avg_redundancy

np.random.seed(SEED)

# Parameters
N = 12                   # candidate documents
M = 4                    # sub-intents
K = QHIVE_PARAMS["K"]

# Simulated relevance & freshness scores
relevance = np.random.rand(N)
freshness = np.random.rand(N)

# Intent lattice
lattice = IntentLattice(m=M, seed=SEED)
intents = lattice.encode_candidates(N)

# Simulated embeddings for redundancy calculation
embeddings = np.random.rand(N, 128)
redundancy = lattice.compute_redundancy(embeddings, intents)

# Run optimizer
optimizer = QMMROptimizer(QHIVE_PARAMS)
selected = optimizer.select(relevance, redundancy, freshness)

print("Selected document indices:", selected)

# Compute metrics
sel_intents = intents[selected]
print("NDCG@K:", ndcg_at_k(relevance[selected], k=K))
print("Intent Coverage:", intent_coverage(sel_intents, M))
print("Average Redundancy:", avg_redundancy(selected, redundancy))
