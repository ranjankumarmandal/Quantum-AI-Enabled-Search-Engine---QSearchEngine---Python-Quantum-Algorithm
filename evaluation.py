"""
Evaluation Metrics for Q-HIVE
-----------------------------
Includes NDCG, intent coverage, redundancy, freshness.
"""

import numpy as np

def ndcg_at_k(relevances, k=10):
    rel = np.array(relevances)[:k]
    dcg = np.sum((2**rel - 1) / np.log2(np.arange(2, len(rel) + 2)))
    ideal = np.sort(rel)[::-1]
    idcg = np.sum((2**ideal - 1) / np.log2(np.arange(2, len(ideal) + 2)))
    return dcg / (idcg + 1e-8)

def intent_coverage(selected_intents, m):
    """
    Fraction of sub-intents covered among selected docs
    """
    covered = set(np.argmax(selected_intents, axis=1))
    return len(covered) / m

def avg_redundancy(selected_idxs, red_matrix):
    """
    Average pairwise redundancy among selected docs
    """
    sel = selected_idxs
    if len(sel) < 2:
        return 0
    vals = []
    for i in range(len(sel)):
        for j in range(i + 1, len(sel)):
            vals.append(red_matrix[sel[i], sel[j]])
    return np.mean(vals)
