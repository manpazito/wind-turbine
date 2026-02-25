from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np


@dataclass
class PolynomialSurrogate:
    """Polynomial-regression surrogate for Cp prediction from BEM samples."""

    degree: int = 2
    terms: list[tuple[int, ...]] = field(default_factory=list)
    coefficients: np.ndarray | None = None

    def _build_terms(self, n_features: int) -> list[tuple[int, ...]]:
        terms: list[tuple[int, ...]] = [tuple()]
        for order in range(1, self.degree + 1):
            terms.extend(combinations_with_replacement(range(n_features), order))
        return terms

    def _transform(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 2:
            raise ValueError("Input must have shape (n_samples, n_features)")
        if not self.terms:
            self.terms = self._build_terms(x.shape[1])

        phi = np.empty((x.shape[0], len(self.terms)), dtype=float)
        for j, term in enumerate(self.terms):
            if len(term) == 0:
                phi[:, j] = 1.0
            else:
                value = np.ones(x.shape[0], dtype=float)
                for idx in term:
                    value *= x[:, idx]
                phi[:, j] = value
        return phi

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must contain the same number of samples")
        phi = self._transform(x)
        coeffs, *_ = np.linalg.lstsq(phi, y, rcond=None)
        self.coefficients = coeffs

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coefficients is None:
            raise RuntimeError("Model must be fit before calling predict")
        phi = self._transform(x)
        return phi @ self.coefficients

    def save(self, path: Path) -> None:
        if self.coefficients is None:
            raise RuntimeError("Cannot save an unfitted model")
        payload = {
            "degree": self.degree,
            "terms": [list(term) for term in self.terms],
            "coefficients": self.coefficients.tolist(),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "PolynomialSurrogate":
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = cls(degree=int(payload["degree"]))
        model.terms = [tuple(term) for term in payload["terms"]]
        model.coefficients = np.asarray(payload["coefficients"], dtype=float)
        return model
