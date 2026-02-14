"""
First-order Takagi-Sugeno-Kang (TSK) Fuzzy Inference System.

A first-order TSK rule has the form:

    R_k : IF x1 is A1k AND x2 is A2k AND ... AND xn is Ank
          THEN y_k = p0k + p1k*x1 + p2k*x2 + ... + pnk*xn

The defuzzified output is computed as a weighted average:

    y = Sum_k (w_k * y_k) / Sum_k (w_k)

where w_k = Product_j mu_{A_jk}(x_j) (product t-norm).
"""

import numpy as np
from membership_functions import gaussian_mf


class TSKRule:
    """
    One TSK rule.

    Attributes:
        antecedent_centres : (n_inputs,) - Gaussian MF centres for each input
        antecedent_sigmas  : (n_inputs,) - Gaussian MF spreads
        consequent_params  : (n_inputs + 1,) - [p0, p1, ..., pn] for first-order TSK
    """

    def __init__(self, centres, sigmas, n_inputs):
        self.antecedent_centres = centres.copy()
        self.antecedent_sigmas = sigmas.copy()
        # Initialise consequent to small random values
        rng = np.random.default_rng(seed=None)
        self.consequent_params = rng.normal(0, 0.1, size=n_inputs + 1)

    def firing_strength(self, X):
        """
        Compute the firing strength for every sample in X using product t-norm.

        Parameters:
            X : (N, n_inputs)

        Returns:
            w : (N,) - firing strengths
        """
        N = X.shape[0]
        w = np.ones(N)
        for j in range(X.shape[1]):
            mu_j = gaussian_mf(X[:, j], self.antecedent_centres[j],
                               self.antecedent_sigmas[j])
            w *= mu_j
        return w

    def consequent_output(self, X):
        """
        Evaluate the linear consequent: y_k = p0 + p1*x1 + ... + pn*xn.

        Parameters:
            X : (N, n_inputs)

        Returns:
            y_k : (N,)
        """
        ones = np.ones((X.shape[0], 1))
        X_aug = np.hstack([ones, X])
        return X_aug @ self.consequent_params


class TSKSystem:
    """
    Complete first-order TSK fuzzy inference system for a single output.
    """

    def __init__(self):
        self.rules = []

    @property
    def n_rules(self):
        return len(self.rules)

    def add_rule(self, rule):
        self.rules.append(rule)

    def predict(self, X):
        """
        Perform TSK inference (weighted-average defuzzification).

        Parameters:
            X : (N, n_inputs)

        Returns:
            y : (N,)
        """
        N = X.shape[0]
        numerator = np.zeros(N)
        denominator = np.zeros(N)

        for rule in self.rules:
            w = rule.firing_strength(X)
            y_k = rule.consequent_output(X)
            numerator += w * y_k
            denominator += w

        denominator = np.maximum(denominator, 1e-12)
        return numerator / denominator

    def get_all_firing_strengths(self, X):
        """Return (N, K) matrix of firing strengths."""
        N = X.shape[0]
        K = self.n_rules
        W = np.zeros((N, K))
        for k, rule in enumerate(self.rules):
            W[:, k] = rule.firing_strength(X)
        return W

    def fit_consequents_lse(self, X, y):
        """
        Estimate consequent parameters by weighted least-squares.

        The output is:
            y = Sum_k (w_bar_k * (p0k + p1k*x1 + ... + pnk*xn))
        
        This is solved via the pseudo-inverse with ridge regularisation.
        """
        N = X.shape[0]
        n = X.shape[1]
        K = self.n_rules

        W = self.get_all_firing_strengths(X)                    # (N, K)
        W_sum = W.sum(axis=1, keepdims=True)
        W_sum = np.maximum(W_sum, 1e-12)
        W_bar = W / W_sum                                       # normalised

        ones = np.ones((N, 1))
        X_aug = np.hstack([ones, X])                            # (N, n+1)

        # Build the design matrix A of shape (N, K*(n+1))
        A = np.zeros((N, K * (n + 1)))
        for k in range(K):
            A[:, k * (n + 1): (k + 1) * (n + 1)] = W_bar[:, k:k + 1] * X_aug

        # Solve via regularised least-squares (ridge regression)
        lambda_reg = 0.01
        ATA = A.T @ A + lambda_reg * np.eye(A.shape[1])
        ATy = A.T @ y
        P = np.linalg.solve(ATA, ATy)

        # Distribute the parameters back to rules
        for k in range(K):
            self.rules[k].consequent_params = P[k * (n + 1): (k + 1) * (n + 1)]
