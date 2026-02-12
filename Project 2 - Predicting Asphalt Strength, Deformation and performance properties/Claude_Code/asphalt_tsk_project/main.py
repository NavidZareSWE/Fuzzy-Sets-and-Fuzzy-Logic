"""
Main execution script for the Asphalt TSK Fuzzy Prediction System.

This script orchestrates:
    1. Data loading and preprocessing
    2. Train/test splitting
    3. Building and tuning four independent TSK fuzzy systems
       (one per output: Stability, Flow, ITSM20, ITSM30)
    4. Evaluation and reporting of RMSE on train and test sets
    5. Saving results and trained models

Usage:
    python main.py
"""

import os
import sys
import json
import pickle
import numpy as np

from config import OUTPUT_COLUMNS, RANDOM_SEED, CLUSTER_RADIUS
from data_loader import load_raw_data, split_data, DataNormaliser
from training import build_tsk_system, tune_tsk_system
from evaluation import rmse, print_results_table


def main():
    np.random.seed(RANDOM_SEED)

    #  1. Load and split data
    print("[1/5] Loading dataset...")
    df = load_raw_data()
    print(f"      Total samples: {len(df)}")

    X_train_raw, X_test_raw, y_train_raw, y_test_raw = split_data(df)
    print(f"      Train: {X_train_raw.shape[0]},  Test: {X_test_raw.shape[0]}")

    #  2. Normalise
    print("[2/5] Normalising data...")
    normaliser = DataNormaliser()
    normaliser.fit(X_train_raw, y_train_raw)

    X_train = normaliser.transform_X(X_train_raw)
    X_test = normaliser.transform_X(X_test_raw)
    y_train = normaliser.transform_y(y_train_raw)
    y_test = normaliser.transform_y(y_test_raw)

    #  3. Build and tune one TSK system per output
    print("[3/5] Building and tuning TSK systems...")
    systems = {}
    histories = {}

    for idx, output_name in enumerate(OUTPUT_COLUMNS):
        print(f"\n   {output_name} ")
        y_tr_col = y_train[:, idx]
        y_te_col = y_test[:, idx]

        system = build_tsk_system(X_train, y_tr_col, ra=CLUSTER_RADIUS)
        print(f"  Rules identified: {system.n_rules}")

        history = tune_tsk_system(
            system, X_train, y_tr_col, verbose=True
        )

        systems[output_name] = system
        histories[output_name] = history

    #  4. Evaluate
    print("\n[4/5] Evaluating on train and test sets...")
    results = {}

    for idx, output_name in enumerate(OUTPUT_COLUMNS):
        system = systems[output_name]

        # Predictions in normalised space
        y_pred_train_norm = system.predict(X_train)
        y_pred_test_norm = system.predict(X_test)

        # Inverse-transform to original scale for RMSE computation
        y_pred_train_orig = normaliser.output_scalers[output_name].inverse_transform(
            y_pred_train_norm.reshape(-1, 1)
        ).ravel()
        y_pred_test_orig = normaliser.output_scalers[output_name].inverse_transform(
            y_pred_test_norm.reshape(-1, 1)
        ).ravel()

        y_true_train = y_train_raw[:, idx]
        y_true_test = y_test_raw[:, idx]

        train_rmse = rmse(y_true_train, y_pred_train_orig)
        test_rmse = rmse(y_true_test, y_pred_test_orig)

        results[output_name] = (train_rmse, test_rmse)

    print()
    print_results_table(results)

    #  5. Save artefacts
    print("\n[5/5] Saving results...")
    os.makedirs("output", exist_ok=True)

    # Save RMSE results as JSON
    results_json = {
        name: {"RMSE_train": tr, "RMSE_test": te}
        for name, (tr, te) in results.items()
    }
    with open("output/rmse_results.json", "w") as f:
        json.dump(results_json, f, indent=2)

    # Save trained systems
    with open("output/trained_systems.pkl", "wb") as f:
        pickle.dump({"systems": systems, "normaliser": normaliser}, f)

    # Save rule summaries
    with open("output/rule_summary.txt", "w") as f:
        for output_name, system in systems.items():
            f.write(f"{'='*60}\n")
            f.write(
                f"Output: {output_name}  |  Number of rules: {system.n_rules}\n")
            f.write(f"{'='*60}\n")
            for r_idx, rule in enumerate(system.rules):
                f.write(f"\nRule {r_idx + 1}:\n")
                f.write(
                    f"  Antecedent centres: {np.round(rule.antecedent_centres, 5).tolist()}\n")
                f.write(
                    f"  Antecedent sigmas:  {np.round(rule.antecedent_sigmas, 5).tolist()}\n")
                f.write(
                    f"  Consequent params:  {np.round(rule.consequent_params, 5).tolist()}\n")
            f.write("\n")

    print("      Results saved to output/")
    print("Done.")
    return results


if __name__ == "__main__":
    results = main()
