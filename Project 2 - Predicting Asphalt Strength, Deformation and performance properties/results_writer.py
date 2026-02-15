import json
import pickle
import numpy as np
from config import OUTPUT_COLUMNS, INPUT_COLUMNS


def save_rmse_results(results, output_path='output/rmse_results.json'):
    results_json = {
        name: {"RMSE_train": float(tr), "RMSE_test": float(te)}
        for name, (tr, te) in results.items()
    }
    with open(output_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"RMSE results saved to {output_path}")


def save_trained_systems(systems, normaliser, output_path='output/trained_systems.pkl'):
    with open(output_path, "wb") as f:
        pickle.dump({"systems": systems, "normaliser": normaliser}, f)
    print(f"Trained systems saved to {output_path}")


def save_rule_summary(systems, output_path='output/rule_summary.txt'):
    with open(output_path, "w") as f:
        for output_name, system in systems.items():
            f.write(f"{'='*60}\n")
            f.write(
                f"Output: {output_name}  |  Number of rules: {system.n_rules()}\n")
            f.write(f"{'='*60}\n")
            for r_idx, rule in enumerate(system.rules):
                f.write(f"\nRule {r_idx + 1}:\n")
                f.write(
                    f"  Antecedent centers: {np.round(rule.antecedent_centers, 5).tolist()}\n")
                f.write(
                    f"  Antecedent sigmas:  {np.round(rule.antecedent_sigmas, 5).tolist()}\n")
                f.write(
                    f"  Consequent params:  {np.round(rule.consequent_params, 5).tolist()}\n")
            f.write("\n")
    print(f"Rule summary saved to {output_path}")


def save_dataset_statistics(df, X_train_raw, X_test_raw,
                            output_path='output/dataset_statistics.txt'):
    with open(output_path, "w") as f:
        f.write("DATASET STATISTICS\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total samples: {len(df)}\n")
        f.write(f"Training samples: {X_train_raw.shape[0]}\n")
        f.write(f"Test samples: {X_test_raw.shape[0]}\n")
        f.write(f"Number of input variables: {len(INPUT_COLUMNS)}\n")
        f.write(f"Number of output variables: {len(OUTPUT_COLUMNS)}\n")
        f.write("\n")
        f.write("Input Variables:\n")
        for col in INPUT_COLUMNS:
            f.write(f"  - {col}\n")
        f.write("\nOutput Variables:\n")
        for col in OUTPUT_COLUMNS:
            f.write(f"  - {col}\n")
    print(f"Dataset statistics saved to {output_path}")


def save_all_results(results, systems, normaliser, df, X_train_raw, X_test_raw,
                     save_model=False):
    print("\nSaving results...")

    save_rmse_results(results)
    save_rule_summary(systems)
    save_dataset_statistics(df, X_train_raw, X_test_raw)

    if save_model:
        save_trained_systems(systems, normaliser)
    else:
        print("Skipping model pickle file (save_model=False)")

    print("All results saved to output/")
