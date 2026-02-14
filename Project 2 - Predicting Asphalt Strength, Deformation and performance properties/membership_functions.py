"""
Gaussian Membership Functions for Fuzzy Variables.
"""

import numpy as np


def gaussian_mf(x, c, sigma):
    """
    Evaluate the Gaussian membership function.
    
    Parameters:
        x     : array of input values
        c     : centre of the Gaussian
        sigma : spread (standard deviation)
    """
    sigma = max(sigma, 1e-10)  # guard against division by zero
    return np.exp(-0.5 * ((x - c) / sigma) ** 2)


class FuzzyVariable:
    """
    Represents one input variable with its associated linguistic terms.
    Each term is a Gaussian MF stored as (centre, sigma).
    """

    def __init__(self, name):
        self.name = name
        self.terms = []  # list of (label, centre, sigma)

    def add_term(self, label, c, sigma):
        self.terms.append((label, c, sigma))

    def fuzzify(self, x):
        """Return membership degrees for every linguistic term."""
        return {label: gaussian_mf(x, c, s) for label, c, s in self.terms}

    def get_params(self):
        """Return arrays of centres and sigmas."""
        centres = np.array([c for _, c, _ in self.terms])
        sigmas = np.array([s for _, _, s in self.terms])
        return centres, sigmas

    def set_params(self, centres, sigmas):
        """Update MF parameters (used during gradient-descent tuning)."""
        new_terms = []
        for idx, (label, _, _) in enumerate(self.terms):
            new_terms.append(
                (label, float(centres[idx]), float(max(sigmas[idx], 1e-10))))
        self.terms = new_terms
