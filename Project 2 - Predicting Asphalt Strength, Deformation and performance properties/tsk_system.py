import numpy as np
from membership_functions import gaussian_mf


class TSKRule:
    def __init__(self, centers, standard_deviations, num_inputs):
        self.antecedent_centers = centers.copy()
        self.antecedent_sigmas = standard_deviations.copy()

        rng = np.random.default_rng(seed=None)
        self.consequent_params = rng.normal(
            0, 0.1, size=num_inputs + 1)

    def firing_level(self, input_data):
        num_samples = input_data.shape[0]
        firing_levels = np.ones(num_samples)
        for feature_idx in range(input_data.shape[1]):
            membership_value = gaussian_mf(input_data[:, feature_idx],
                                           self.antecedent_centers[feature_idx],
                                           self.antecedent_sigmas[feature_idx])
            firing_levels *= membership_value
        return firing_levels

    def consequent_output(self, input_data):
        bias_column = np.ones((input_data.shape[0], 1))
        augmented_features = np.hstack([bias_column, input_data])
        return augmented_features @ self.consequent_params


class TSKSystem:
    def __init__(self):
        self.rules = []

    def n_rules(self):
        return len(self.rules)

    def add_rule(self, rule):
        self.rules.append(rule)

    def predict(self, input_data):
        num_samples = input_data.shape[0]
        numerator = np.zeros(num_samples)
        denominator = np.zeros(num_samples)

        for rule in self.rules:
            firing_levels = rule.firing_level(input_data)
            rule_output = rule.consequent_output(input_data)
            numerator += firing_levels * rule_output
            denominator += firing_levels

        denominator = np.maximum(denominator, 1e-12)
        return numerator / denominator

    def get_all_firing_levels(self, input_data):
        num_samples = input_data.shape[0]
        num_rules = self.n_rules()
        firing_level_matrix = np.zeros((num_samples, num_rules))
        for rule_idx, rule in enumerate(self.rules):
            firing_level_matrix[:,
                                rule_idx] = rule.firing_level(input_data)
        return firing_level_matrix

    def fit_consequents_lse(self, input_data, target):
        num_samples = input_data.shape[0]
        num_features = input_data.shape[1]
        num_rules = self.n_rules()

        firing_level_matrix = self.get_all_firing_levels(
            input_data)
        total_firing_level = firing_level_matrix.sum(
            axis=1, keepdims=True)
        total_firing_level = np.maximum(total_firing_level, 1e-12)
        normalized_firing_levels = firing_level_matrix / total_firing_level

        bias_column = np.ones((num_samples, 1))
        augmented_features = np.hstack([bias_column, input_data])

        design_matrix = np.zeros((num_samples, num_rules * (num_features + 1)))
        for rule_idx in range(num_rules):
            start_col = rule_idx * (num_features + 1)
            end_col = (rule_idx + 1) * (num_features + 1)
            design_matrix[:, start_col:end_col] = normalized_firing_levels[:,
                                                                           rule_idx:rule_idx + 1] * augmented_features

        # Solve via regularised least-squares (ridge regression)
        regularization_lambda = 0.01
        gram_matrix = design_matrix.T @ design_matrix + \
            regularization_lambda * np.eye(design_matrix.shape[1])
        target_correlation = design_matrix.T @ target
        all_consequent_params = np.linalg.solve(
            gram_matrix, target_correlation)

        # Distribute the parameters back to rules
        for rule_idx in range(num_rules):
            start_idx = rule_idx * (num_features + 1)
            end_idx = (rule_idx + 1) * (num_features + 1)
            self.rules[rule_idx].consequent_params = all_consequent_params[start_idx:end_idx]
