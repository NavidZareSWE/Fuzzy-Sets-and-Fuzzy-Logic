import numpy as np


def gaussian_mf(x, center, sigma):
    sigma = max(sigma, 1e-10)  # Avoid division by zero
    return np.exp(-0.5 * ((x - center) / sigma) ** 2)


class FuzzyVariable:
    def __init__(self, name):
        self.name = name
        # list of (label, center, sigma)
        # ("cold", 0.0, 1.5)
        # ("warm", 5.0, 1.2)
        # ("hot", 9.0, 1.8)
        self.terms = []

    def add_term(self, label, c, sigma):
        self.terms.append((label, c, sigma))

    def fuzzify(self, x):
        return {label: gaussian_mf(x, _center, _sigma) for label, _center, _sigma in self.terms}

    def get_params(self):
        centers = np.array([c for _, c, _ in self.terms])
        sigmas = np.array([s for _, _, s in self.terms])
        return centers, sigmas

    def set_params(self, centers, sigmas):
        new_terms = []
        for idx, (label, _, _) in enumerate(self.terms):
            new_terms.append(
                (label, float(centers[idx]), float(max(sigmas[idx], 1e-10))))
        self.terms = new_terms
