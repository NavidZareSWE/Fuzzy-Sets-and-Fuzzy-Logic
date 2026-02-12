"""
Training pipeline for the TSK Fuzzy System.

The training procedure follows a two-phase hybrid approach as described
in Mendel (2017, §9.6–9.7):

Phase 1 — Structure identification:
    Subtractive clustering on the joint (input, output) space determines
    the number of rules and initialises the antecedent MF parameters.

Phase 2 — Parameter optimisation:
    (a) Consequent parameters are estimated via least-squares estimation
        (LSE) with the antecedent parameters fixed.
    (b) Antecedent MF parameters (centres and spreads) are then fine-tuned
        using gradient descent to minimise the mean squared error,
        while consequent parameters are re-estimated by LSE at each epoch.

This hybrid strategy is analogous to the ANFIS learning algorithm
(Jang, 1993), which Mendel discusses as a canonical example of
TSK system tuning.
"""

import numpy as np
from clustering import subtractive_clustering
from tsk_system import TSKSystem, TSKRule
from config import (
    LEARNING_RATE, MAX_EPOCHS, TOLERANCE, CLUSTER_RADIUS,
)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def build_tsk_system(
    X_train: np.ndarray,
    y_train_col: np.ndarray,
    ra: float = CLUSTER_RADIUS,
) -> TSKSystem:
    """
    Build a TSK system for one output variable.

    Steps
    -----
    1. Concatenate normalised inputs and the target output column.
    2. Run subtractive clustering to discover rule prototypes.
    3. Initialise one TSK rule per cluster centre.
    4. Estimate consequent parameters via LSE.
    """
    n_inputs = X_train.shape[1]

    # Joint space for clustering
    joint = np.hstack([X_train, y_train_col.reshape(-1, 1)])
    centres = subtractive_clustering(joint, ra=ra)

    system = TSKSystem()
    for centre in centres:
        # Antecedent centres are the input-space projection of the cluster centre
        c_input = centre[:n_inputs]
        # Spread is estimated as a fraction of the cluster radius
        sigma_input = np.full(n_inputs, ra / np.sqrt(8.0))
        rule = TSKRule(c_input, sigma_input, n_inputs)
        system.add_rule(rule)

    # Initial LSE for consequent parameters
    system.fit_consequents_lse(X_train, y_train_col)

    return system


def tune_tsk_system(
    system: TSKSystem,
    X_train: np.ndarray,
    y_train_col: np.ndarray,
    lr: float = LEARNING_RATE,
    max_epochs: int = MAX_EPOCHS,
    tol: float = TOLERANCE,
    verbose: bool = False,
) -> list[float]:
    """
    Fine-tune antecedent parameters via gradient descent and
    re-estimate consequents via LSE at each epoch (hybrid learning).

    The gradient of the MSE with respect to each antecedent centre c_{jk}
    and spread sigma_{jk} is computed analytically.

    Returns a list of RMSE values per epoch for convergence monitoring.
    """
    N = X_train.shape[0]
    n = X_train.shape[1]
    K = system.n_rules
    history = []

    for epoch in range(max_epochs):
        #  Forward pass
        y_pred = system.predict(X_train)
        error = y_train_col - y_pred
        rmse_val = _rmse(y_train_col, y_pred)
        history.append(rmse_val)

        if verbose and (epoch % 50 == 0 or epoch == max_epochs - 1):
            print(f"  Epoch {epoch:4d}  RMSE = {rmse_val:.6f}")

        if epoch > 0 and abs(history[-2] - history[-1]) < tol:
            if verbose:
                print(f"  Converged at epoch {epoch}.")
            break

        #  Backward pass: gradient w.r.t. antecedent parameters
        W = system.get_all_firing_strengths(X_train)        # (N, K)
        W_sum = W.sum(axis=1, keepdims=True)
        W_sum = np.maximum(W_sum, 1e-12)
        W_bar = W / W_sum                                   # (N, K)

        # Consequent outputs per rule
        Y_k = np.zeros((N, K))
        for k in range(K):
            Y_k[:, k] = system.rules[k].consequent_output(X_train)

        for k in range(K):
            rule = system.rules[k]
            w_k = W[:, k]                                    # (N,)
            w_bar_k = W_bar[:, k]                            # (N,)
            y_k = Y_k[:, k]                                  # (N,)

            # dE/d(w_bar_k) = -2/N * error * (y_k - y_pred)
            # dw_bar_k/dw_k = (W_sum.ravel() - w_k) / (W_sum.ravel() ** 2)
            dE_dwbar = -error * (y_k - y_pred)               # (N,)
            dwbar_dw = (W_sum.ravel() - w_k) / (W_sum.ravel() ** 2)  # (N,)
            dE_dw = dE_dwbar * dwbar_dw                      # (N,)

            for j in range(n):
                x_j = X_train[:, j]
                c_jk = rule.antecedent_centres[j]
                s_jk = rule.antecedent_sigmas[j]
                s_jk = max(s_jk, 1e-10)

                # dw_k/dc_jk = w_k * (x_j - c_jk) / s_jk^2
                dw_dc = w_k * (x_j - c_jk) / (s_jk ** 2)
                grad_c = np.mean(dE_dw * dw_dc)

                # dw_k/ds_jk = w_k * (x_j - c_jk)^2 / s_jk^3
                dw_ds = w_k * ((x_j - c_jk) ** 2) / (s_jk ** 3)
                grad_s = np.mean(dE_dw * dw_ds)

                rule.antecedent_centres[j] -= lr * grad_c
                new_sigma = rule.antecedent_sigmas[j] - lr * grad_s
                rule.antecedent_sigmas[j] = max(new_sigma, 1e-6)

        # Re-estimate consequents by LSE after antecedent update
        system.fit_consequents_lse(X_train, y_train_col)

    return history
