import numpy as np
from clustering import subtractive_clustering
from tsk_system import TSKSystem, TSKRule
from config import LEARNING_RATE, MAX_EPOCHS, TOLERANCE, CLUSTER_RADIUS


def _rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def build_tsk_system(X_train, y_train_col, cluster_radius=CLUSTER_RADIUS):
    n_dimesnsions = X_train.shape[1]

    # Joint space for clustering
    # So instead of clustering only in input space, you cluster in (X, y) space.
    # This forces clusters to group points that are:
    # close in input space and
    # produce similar outputs
    joint = np.hstack([X_train, y_train_col.reshape(-1, 1)])
    centers = subtractive_clustering(joint, cluster_radius=cluster_radius)

    system = TSKSystem()
    for center in centers:
        # Drop the last dimension (the output) → keep only the input part.
        c_input = center[:n_dimesnsions]
        # Spread is estimated as a fraction of the cluster radius
        # standard heuristic from subtractive clustering.
        sigma_input = np.full(n_dimesnsions, cluster_radius / np.sqrt(8.0))
        rule = TSKRule(c_input, sigma_input, n_dimesnsions)
        system.add_rule(rule)

    system.fit_consequents_lse(X_train, y_train_col)

    return system


def tune_tsk_system(system, X_train, y_train_col,
                    lr=LEARNING_RATE,
                    max_epochs=MAX_EPOCHS,
                    tol=TOLERANCE):

    n_samples, n_features = X_train.shape
    n_rules = system.n_rules
    rmse_history = []

    for epoch in range(max_epochs):

        # ---------- Forward pass ----------
        y_pred = system.predict(X_train)
        error = y_train_col - y_pred
        rmse_val = _rmse(y_train_col, y_pred)
        rmse_history.append(rmse_val)

        print(f"Epoch {epoch}: RMSE = {rmse_val:.6f}")

        # Stops early if improvement is below tol
        if epoch > 0 and abs(rmse_history[-2] - rmse_history[-1]) < tol:
            break

        # ---------- Backward pass ----------
        firing_levels = system.get_all_firing_levels(
            X_train)
        # keepdimsbool, optional If this is set to True, the axes which are reduced are left in the result as dimensions with size one. With this option, the result will broadcast correctly against the input array.
        # Read More: https://numpy.org/doc/2.1/reference/generated/numpy.sum.html
        firing_sum = firing_levels.sum(axis=1, keepdims=True)
        firing_sum = np.maximum(firing_sum, 1e-12)
        normalized_firing = firing_levels / \
            firing_sum

        # ---------- Rule outputs ----------
        rule_outputs = np.zeros((n_samples, n_rules))
        for r in range(n_rules):
            rule_outputs[:, r] = system.rules[r].consequent_output(X_train)

        # ---------- Antecedent gradient descent ----------
        for r in range(n_rules):
            rule = system.rules[r]

            rule_firing_level = firing_levels[:, r]
            rule_output_values = rule_outputs[:, r]

            # ∂E/∂(normalized firing): how the error changes w.r.t the normalized firing of this rule
            dE_dwbar = -error * (rule_output_values - y_pred)

            # ∂(normalized firing)/∂(raw firing): how normalized firing changes w.r.t the raw firing
            d_wbar_d_w = (firing_sum.ravel() - rule_firing_level) / \
                (firing_sum.ravel() ** 2)

            # ∂E/∂(raw firing): chain rule to get error gradient w.r.t raw firing strength
            dE_dw = dE_dwbar * d_wbar_d_w

            for f in range(n_features):
                x_f = X_train[:, f]
                center_rf = rule.antecedent_centers[f]
                sigma_rf = max(rule.antecedent_sigmas[f], 1e-10)

                # ∂(firing)/∂(center): how firing changes w.r.t antecedent center
                d_w_d_c = rule_firing_level * \
                    (x_f - center_rf) / (sigma_rf ** 2)
                grad_center = np.mean(dE_dw * d_w_d_c)

                # ∂(firing)/∂(sigma): how firing changes w.r.t antecedent sigma
                d_w_d_s = rule_firing_level * \
                    ((x_f - center_rf) ** 2) / (sigma_rf ** 3)
                grad_sigma = np.mean(dE_dw * d_w_d_s)

                # Update antecedent parameters using gradient descent
                rule.antecedent_centers[f] -= lr * grad_center
                rule.antecedent_sigmas[f] = max(
                    rule.antecedent_sigmas[f] - lr * grad_sigma, 1e-6
                )

        # ---------- Re-fit consequents ----------
        system.fit_consequents_lse(X_train, y_train_col)

    return rmse_history
