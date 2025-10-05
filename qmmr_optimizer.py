"""
Q-MMR Optimizer
---------------
Selects top-K documents maximizing relevance and diversity.
"""

import numpy as np
from typing import List, Tuple
from quantum.config import QHIVE_PARAMS

try:
    from qiskit.algorithms import QAOA
    from qiskit_optimization import QuadraticProgram
    from qiskit_optimization.algorithms import MinimumEigenOptimizer
    from qiskit_optimization.converters import QuadraticProgramToQubo
    from qiskit.primitives import Estimator
except ImportError:
    QAOA = None  # if Qiskit not installed


class QMMROptimizer:
    def __init__(self, params: dict = QHIVE_PARAMS):
        self.K = params["K"]
        self.lam = params["lambda_"]
        self.mu = params["mu"]
        self.gamma = params["gamma"]
        self.use_qaoa = params["use_qaoa"]
        self.qaoa_reps = params["qaoa_reps"]

    def build_objective(self, rel: np.ndarray,
                        red: np.ndarray,
                        fresh: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = len(rel)
        a = rel + self.gamma * fresh - self.mu
        b = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                b[i, j] = self.lam * red[i, j] + self.mu
        return a, b

    def solve_classical(self, a: np.ndarray, b: np.ndarray) -> List[int]:
        n = len(a)
        x = np.zeros(n, dtype=int)
        # greedy init
        top_idx = np.argsort(a)[::-1][:self.K]
        x[top_idx] = 1

        # local improvement
        improved = True
        while improved:
            improved = False
            for i in range(n):
                x_flip = x.copy()
                x_flip[i] = 1 - x_flip[i]
                gain = self.objective(x_flip, a, b) - self.objective(x, a, b)
                if gain > 1e-6:
                    x = x_flip
                    improved = True
        return x.tolist()

    def objective(self, x: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, x) - np.sum(b * np.outer(x, x))

    def solve_qaoa(self, a: np.ndarray, b: np.ndarray) -> List[int]:
        if QAOA is None:
            print("Qiskit not installed, using classical solver.")
            return self.solve_classical(a, b)

        n = len(a)
        qp = QuadraticProgram()
        for i in range(n):
            qp.binary_var(f"x{i}")

        linear = {f"x{i}": -a[i] for i in range(n)}
        quadratic = {(f"x{i}", f"x{j}"): b[i, j] for i in range(n) for j in range(i+1, n)}
        qp.minimize(linear=linear, quadratic=quadratic)

        qubo = QuadraticProgramToQubo().convert(qp)
        solver = MinimumEigenOptimizer(QAOA(estimator=Estimator(), reps=self.qaoa_reps))
        result = solver.solve(qubo)
        return [int(result.x[i]) for i in range(n)]

    def select(self, rel, red, fresh):
        a, b = self.build_objective(rel, red, fresh)
        if self.use_qaoa:
            print("Running QAOA solver...")
            return self.solve_qaoa(a, b)
        else:
            print("Running classical solver...")
            return self.solve_classical(a, b)
