import numpy as np
import matplotlib.pyplot as plt
from config import OUTPUT_COLUMNS, INPUT_COLUMNS


def plot_training_convergence(histories, output_path='output/training_convergence.png'):

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for idx, (output_name, history) in enumerate(histories.items()):
        ax = axes[idx // 2, idx % 2]
        ax.plot(history, 'b-', linewidth=1)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE (Normalised)')
        ax.set_title(f'Training Convergence - {output_name}')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_predicted_vs_actual_test(predictions, output_path='output/predicted_vs_actual_test.png'):

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    units = {'Stability': 'kN', 'Flow': 'mm', 'ITSM20': 'MPa', 'ITSM30': 'MPa'}

    for idx, output_name in enumerate(OUTPUT_COLUMNS):
        ax = axes[idx // 2, idx % 2]
        y_true = predictions[output_name]['test_true']
        y_pred = predictions[output_name]['test_pred']

        ax.scatter(y_true, y_pred, alpha=0.7,
                   edgecolors='black', linewidth=0.5)

        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val],
                'r--', label='Perfect Prediction')

        ax.set_xlabel(f'Actual {output_name} ({units[output_name]})')
        ax.set_ylabel(f'Predicted {output_name} ({units[output_name]})')
        ax.set_title(f'{output_name} - Test Set')
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_predicted_vs_actual_train(predictions, output_path='output/predicted_vs_actual_train.png'):

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    units = {'Stability': 'kN', 'Flow': 'mm', 'ITSM20': 'MPa', 'ITSM30': 'MPa'}

    for idx, output_name in enumerate(OUTPUT_COLUMNS):
        ax = axes[idx // 2, idx % 2]
        y_true = predictions[output_name]['train_true']
        y_pred = predictions[output_name]['train_pred']

        ax.scatter(y_true, y_pred, alpha=0.7,
                   edgecolors='black', linewidth=0.5)

        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val],
                'r--', label='Perfect Prediction')

        ax.set_xlabel(f'Actual {output_name} ({units[output_name]})')
        ax.set_ylabel(f'Predicted {output_name} ({units[output_name]})')
        ax.set_title(f'{output_name} - Training Set')
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_error_distribution_test(predictions, output_path='output/error_distribution_test.png'):

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    units = {'Stability': 'kN', 'Flow': 'mm', 'ITSM20': 'MPa', 'ITSM30': 'MPa'}

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
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_membership_functions(system, output_name='Stability',
                              output_path='output/membership_functions_stability.png'):

    n_rules = system.n_rules
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    x_range = np.linspace(0, 1, 200)

    for j, input_name in enumerate(INPUT_COLUMNS):
        ax = axes[j // 5, j % 5]
        for r_idx, rule in enumerate(system.rules):
            c = rule.antecedent_centers[j]
            s = rule.antecedent_sigmas[j]
            mf_values = np.exp(-0.5 * ((x_range - c) / s) ** 2)
            ax.plot(x_range, mf_values, label=f'Rule {r_idx+1}')

        ax.set_xlabel(f'{input_name} (normalised)')
        ax.set_ylabel('Membership Degree')
        ax.set_title(f'{input_name}')
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        f'Membership Functions for {output_name} System ({n_rules} Rules)',
        fontsize=14
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_rmse_comparison(results, output_path='output/rmse_comparison.png'):

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(OUTPUT_COLUMNS))
    width = 0.35

    train_rmse = [results[name][0] for name in OUTPUT_COLUMNS]
    test_rmse = [results[name][1] for name in OUTPUT_COLUMNS]

    bars1 = ax.bar(x - width/2, train_rmse, width,
                   label='Train RMSE', color='steelblue')
    bars2 = ax.bar(x + width/2, test_rmse, width,
                   label='Test RMSE', color='coral')

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
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def generate_all_plots(histories, predictions, results, systems):
    print("\nGenerating plots...")

    plot_training_convergence(histories)
    plot_predicted_vs_actual_test(predictions)
    plot_predicted_vs_actual_train(predictions)
    plot_error_distribution_test(predictions)
    plot_membership_functions(systems['Stability'], output_name='Stability')
    plot_rmse_comparison(results)

    print("All plots saved to output/")
