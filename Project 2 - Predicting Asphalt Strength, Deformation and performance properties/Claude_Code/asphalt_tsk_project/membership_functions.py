"""
Fuzzy membership functions.

Gaussian membership functions are used following the formulation in
Mendel (2017), Chapter 2.  Each linguistic term is characterised by
a centre (mean) c and a spread (standard deviation) sigma:

    mu(x) = exp( -0.5 * ((x - c) / sigma)^2 )

Klir & Yuan (1995, §2.3) provide the general mathematical basis for
fuzzy sets and membership functions on which this implementation relies.
"""

import numpy as np


def gaussian_mf(x: np.ndarray, c: float, sigma: float) -> np.ndarray:
    """
    Evaluate the Gaussian membership function.

    Parameters
    ----------
    x     : array of input values
    c     : centre of the Gaussian
    sigma : spread (standard deviation); must be > 0
    """
    sigma = max(sigma, 1e-10)  # guard against division by zero
    return np.exp(-0.5 * ((x - c) / sigma) ** 2)


class FuzzyVariable:
    """
    Represents one input variable with its associated linguistic terms.

    Each term is a Gaussian MF stored as (centre, sigma).
    """

    def __init__(self, name: str):
        self.name = name
        self.terms: list[tuple[str, float, float]] = []  # (label, c, sigma)

    def add_term(self, label: str, c: float, sigma: float):
        self.terms.append((label, c, sigma))

    def fuzzify(self, x: np.ndarray) -> dict[str, np.ndarray]:
        """Return membership degrees for every linguistic term."""
        return {label: gaussian_mf(x, c, s) for label, c, s in self.terms}

    def get_params(self):
        """Return flat arrays of centres and sigmas for optimisation."""
        centres = np.array([c for _, c, _ in self.terms])
        sigmas = np.array([s for _, _, s in self.terms])
        return centres, sigmas

    def set_params(self, centres: np.ndarray, sigmas: np.ndarray):
        """Update MF parameters (used during gradient-descent tuning)."""
        new_terms = []
        for idx, (label, _, _) in enumerate(self.terms):
            new_terms.append((label, float(centres[idx]), float(max(sigmas[idx], 1e-10))))
        self.terms = new_terms

    def __repr__(self):
        parts = ", ".join(f"{l}(c={c:.4f}, σ={s:.4f})" for l, c, s in self.terms)
        return f"FuzzyVariable({self.name}: {parts})"
