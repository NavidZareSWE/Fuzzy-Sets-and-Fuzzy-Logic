"""
Main execution script for the Asphalt TSK Fuzzy Prediction System.

This script:
    1. Loads and preprocesses data
    2. Builds and tunes four independent TSK fuzzy systems
    3. Evaluates and reports RMSE on train and test sets
    4. Generates plots and saves results for the report
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json

from config import OUTPUT_COLUMNS, RANDOM_SEED, CLUSTER_RADIUS, INPUT_COLUMNS
from data_loader import load_raw_data, split_data, DataNormaliser
from training import build_tsk_system, tune_tsk_system
from evaluation import rmse


def main():
    np.random.seed(RANDOM_SEED)
    os.makedirs("output", exist_ok=True)

    # 1. Load and split data
    df = load_raw_data()
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = split_data(df)

    # 2. Normalise
    normaliser = DataNormaliser()
    normaliser.fit(X_train_raw, y_train_raw)

    X_train = normaliser.transform_X(X_train_raw)
    X_test = normaliser.transform_X(X_test_raw)
    y_train = normaliser.transform_y(y_train_raw)
    y_test = normaliser.transform_y(y_test_raw)

    # 3. Build and tune one TSK system per output
    systems = {}
    histories = {}

    for idx, output_name in enumerate(OUTPUT_COLUMNS):
        y_tr_col = y_train[:, idx]
        system = build_tsk_system(X_train, y_tr_col, ra=CLUSTER_RADIUS)
        history = tune_tsk_system(system, X_train, y_tr_col, verbose=False)
        systems[output_name] = system
        histories[output_name] = history

    # 4. Evaluate
    results = {}
    predictions = {}

    for idx, output_name in enumerate(OUTPUT_COLUMNS):
        system = systems[output_name]

        # Predictions in normalised space
        y_pred_train_norm = system.predict(X_train)
        y_pred_test_norm = system.predict(X_test)

        # Inverse-transform to original scale
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
        predictions[output_name] = {
            'train_true': y_true_train,
            'train_pred': y_pred_train_orig,
            'test_true': y_true_test,
            'test_pred': y_pred_test_orig
        }

    # Print RMSE Results Table
    print("\n" + "=" * 50)
    print("RMSE RESULTS")
    print("=" * 50)
    print(f"{'Output':<12} {'RMSE(Train)':>14} {'RMSE(Test)':>14}")
    print("-" * 50)
    for name, (tr, te) in results.items():
        print(f"{name:<12} {tr:>14.4f} {te:>14.4f}")
    print("=" * 50)

    # Print Number of Rules
    print("\n" + "=" * 50)
    print("NUMBER OF RULES PER OUTPUT")
    print("=" * 50)
    for name, system in systems.items():
        print(f"{name:<12}: {system.n_rules} rules")
    print("=" * 50)

    # 5. Generate plots for the report

    # Plot 1: Training Convergence
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for idx, (output_name, history) in enumerate(histories.items()):
        ax = axes[idx // 2, idx % 2]
        ax.plot(history, 'b-', linewidth=1)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE (Normalised)')
        ax.set_title(f'Training Convergence - {output_name}')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('output/training_convergence.png', dpi=150)
    plt.close()

    # Plot 2: Predicted vs Actual (Test Set)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    units = {'Stability': 'kN', 'Flow': 'mm', 'ITSM20': 'MPa', 'ITSM30': 'MPa'}
    for idx, output_name in enumerate(OUTPUT_COLUMNS):
        ax = axes[idx // 2, idx % 2]
        y_true = predictions[output_name]['test_true']
        y_pred = predictions[output_name]['test_pred']
        
        ax.scatter(y_true, y_pred, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        ax.set_xlabel(f'Actual {output_name} ({units[output_name]})')
        ax.set_ylabel(f'Predicted {output_name} ({units[output_name]})')
        ax.set_title(f'{output_name} - Test Set')
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('output/predicted_vs_actual_test.png', dpi=150)
    plt.close()

    # Plot 3: Predicted vs Actual (Train Set)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for idx, output_name in enumerate(OUTPUT_COLUMNS):
        ax = axes[idx // 2, idx % 2]
        y_true = predictions[output_name]['train_true']
        y_pred = predictions[output_name]['train_pred']
        
        ax.scatter(y_true, y_pred, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        ax.set_xlabel(f'Actual {output_name} ({units[output_name]})')
        ax.set_ylabel(f'Predicted {output_name} ({units[output_name]})')
        ax.set_title(f'{output_name} - Training Set')
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('output/predicted_vs_actual_train.png', dpi=150)
    plt.close()

    # Plot 4: Error Distribution (Test Set)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for idx, output_name in enumerate(OUTPUT_COLUMNS):
        ax = axes[idx // 2, idx % 2]
        y_true = predictions[output_name]['test_true']
        y_pred = predictions[output_name]['test_pred']
        errors = y_pred - y_true
        
        ax.hist(errors, bins=15, edgecolor='black', alpha=0.7)
        ax.axvline(x=0, color='r', linestyle='--', label='Zero Error')
        ax.set_xlabel(f'Prediction Error ({units[output_name]})')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{output_name} - Error Distribution (Test Set)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('output/error_distribution_test.png', dpi=150)
    plt.close()

    # Plot 5: Membership Functions for one output (Stability)
    output_name = 'Stability'
    system = systems[output_name]
    n_rules = system.n_rules
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    x_range = np.linspace(0, 1, 200)
    
    for j, input_name in enumerate(INPUT_COLUMNS):
        ax = axes[j // 5, j % 5]
        for r_idx, rule in enumerate(system.rules):
            c = rule.antecedent_centres[j]
            s = rule.antecedent_sigmas[j]
            mf_values = np.exp(-0.5 * ((x_range - c) / s) ** 2)
            ax.plot(x_range, mf_values, label=f'Rule {r_idx+1}')
        
        ax.set_xlabel(f'{input_name} (normalised)')
        ax.set_ylabel('Membership Degree')
        ax.set_title(f'{input_name}')
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Membership Functions for {output_name} System ({n_rules} Rules)', fontsize=14)
    plt.tight_layout()
    plt.savefig('output/membership_functions_stability.png', dpi=150)
    plt.close()

    # Plot 6: RMSE Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(OUTPUT_COLUMNS))
    width = 0.35
    
    train_rmse = [results[name][0] for name in OUTPUT_COLUMNS]
    test_rmse = [results[name][1] for name in OUTPUT_COLUMNS]
    
    bars1 = ax.bar(x - width/2, train_rmse, width, label='Train RMSE', color='steelblue')
    bars2 = ax.bar(x + width/2, test_rmse, width, label='Test RMSE', color='coral')
    
    ax.set_xlabel('Output Variable')
    ax.set_ylabel('RMSE')
    ax.set_title('RMSE Comparison: Training vs Test Data')
    ax.set_xticks(x)
    ax.set_xticklabels(OUTPUT_COLUMNS)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('output/rmse_comparison.png', dpi=150)
    plt.close()

    # 6. Save artefacts

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
            f.write(f"Output: {output_name}  |  Number of rules: {system.n_rules}\n")
            f.write(f"{'='*60}\n")
            for r_idx, rule in enumerate(system.rules):
                f.write(f"\nRule {r_idx + 1}:\n")
                f.write(f"  Antecedent centres: {np.round(rule.antecedent_centres, 5).tolist()}\n")
                f.write(f"  Antecedent sigmas:  {np.round(rule.antecedent_sigmas, 5).tolist()}\n")
                f.write(f"  Consequent params:  {np.round(rule.consequent_params, 5).tolist()}\n")
            f.write("\n")

    # Save dataset statistics
    with open("output/dataset_statistics.txt", "w") as f:
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

    print("\nPlots and results saved to output/")
    return results


if __name__ == "__main__":
    results = main()
