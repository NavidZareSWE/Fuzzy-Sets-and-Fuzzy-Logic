"""
Training pipeline for the TSK Fuzzy System.

The training procedure follows a two-phase hybrid approach:

Phase 1 - Structure identification:
    Subtractive clustering on the joint (input, output) space determines
    the number of rules and initialises the antecedent MF parameters.

Phase 2 - Parameter optimisation:
    (a) Consequent parameters are estimated via least-squares estimation
        (LSE) with the antecedent parameters fixed.
    (b) Antecedent MF parameters (centres and spreads) are then fine-tuned
        using gradient descent to minimise the mean squared error,
        while consequent parameters are re-estimated by LSE at each epoch.
"""

import numpy as np
from clustering import subtractive_clustering
from tsk_system import TSKSystem, TSKRule
from config import LEARNING_RATE, MAX_EPOCHS, TOLERANCE, CLUSTER_RADIUS


def _rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def build_tsk_system(X_train, y_train_col, ra=CLUSTER_RADIUS):
    """
    Build a TSK system for one output variable.

    Steps:
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


def tune_tsk_system(system, X_train, y_train_col, lr=LEARNING_RATE,
                    max_epochs=MAX_EPOCHS, tol=TOLERANCE, verbose=False):
    """
    Fine-tune antecedent parameters via gradient descent and
    re-estimate consequents via LSE at each epoch (hybrid learning).

    Returns a list of RMSE values per epoch for convergence monitoring.
    """
    N = X_train.shape[0]
    n = X_train.shape[1]
    K = system.n_rules
    history = []

    for epoch in range(max_epochs):
        # Forward pass
        y_pred = system.predict(X_train)
        error = y_train_col - y_pred
        rmse_val = _rmse(y_train_col, y_pred)
        history.append(rmse_val)

        if epoch > 0 and abs(history[-2] - history[-1]) < tol:
            break

        # Backward pass: gradient w.r.t. antecedent parameters
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
            w_k = W[:, k]
            w_bar_k = W_bar[:, k]
            y_k = Y_k[:, k]

            # Gradient calculations
            dE_dwbar = -error * (y_k - y_pred)
            dwbar_dw = (W_sum.ravel() - w_k) / (W_sum.ravel() ** 2)
            dE_dw = dE_dwbar * dwbar_dw

            for j in range(n):
                x_j = X_train[:, j]
                c_jk = rule.antecedent_centres[j]
                s_jk = max(rule.antecedent_sigmas[j], 1e-10)

                # Gradient for centre
                dw_dc = w_k * (x_j - c_jk) / (s_jk ** 2)
                grad_c = np.mean(dE_dw * dw_dc)

                # Gradient for sigma
                dw_ds = w_k * ((x_j - c_jk) ** 2) / (s_jk ** 3)
                grad_s = np.mean(dE_dw * dw_ds)

                rule.antecedent_centres[j] -= lr * grad_c
                new_sigma = rule.antecedent_sigmas[j] - lr * grad_s
                rule.antecedent_sigmas[j] = max(new_sigma, 1e-6)

        # Re-estimate consequents by LSE after antecedent update
        system.fit_consequents_lse(X_train, y_train_col)

    return history
