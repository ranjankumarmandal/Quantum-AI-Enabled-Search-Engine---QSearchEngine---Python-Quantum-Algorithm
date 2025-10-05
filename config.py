"""
Q-HIVE Configurations
---------------------
Holds tunable parameters and environment settings.
"""

QHIVE_PARAMS = {
    "K": 10,                     # documents to select
    "lambda_": 0.5,               # relevance vs. redundancy trade-off
    "mu": 1.0,                    # fixed-K constraint penalty
    "gamma": 0.1,                 # freshness / trust weight
    "qaoa_reps": 3,               # QAOA circuit depth
    "use_qaoa": False             # set True to use quantum solver
}

SEED = 42
