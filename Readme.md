# Quantum AI Enabled Search Engine (Python Quantum Algorithm Component)

Python Quantum Component:

This module provides the quantum-AI layer for the Q-HIVE search engine. It implements the Q-MMR (Quantum-Inspired Max-Marginal-Relevance) optimizer, which performs global, diversity-aware document re-ranking using either a classical Ising solver or a QAOA-based quantum optimizer.

## ⚙️ Requirements

Python ≥ 3.9, NumPy ≥ 1.23, Qiskit ≥ 0.45 (optional, only needed for QAOA solver)

## Install dependencies

pip install libs-above

## 📊 Features

- Builds Ising-formulated objective from relevance, redundancy, and freshness signals.
- Optimizes using:
- - Classical heuristic/annealing solver (production-ready, low-latency).
- - QAOA-based solver (quantum embodiment for research).
- Integrates seamlessly with Java-based retrieval pipeline via JSON or CSV I/O.
- Includes standard evaluation metrics:
- - NDCG@K
- - Intent Coverage
- - Average Redundancy
- Fully reproducible demo with random seed and dummy data.

## You have some query?

If you have some query, feel free to connect with me here -- [Ranjan Kumar Mandal](https://www.linkedin.com/in/ranjan-kumar-m-818367158/)
