import numpy as np
import matplotlib.pyplot as plt
from config import OUTPUT_COLUMNS, INPUT_COLUMNS


def plot_training_convergence(histories, output_path='output/training_convergence.png'):
    """
    Shows RMSE decrease over epochs during gradient descent tuning.
    Purpose: Demonstrates that the TSK system parameters were successfully tuned.
    """
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
    """
    Scatter plot of predicted vs actual values on TEST set.
    Purpose: Shows model generalization ability on unseen data.
    Points close to diagonal line = good predictions.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    units = {'Stability': 'kN', 'Flow': 'mm', 'ITSM20': 'MPa', 'ITSM30': 'MPa'}

    for idx, output_name in enumerate(OUTPUT_COLUMNS):
        ax = axes[idx // 2, idx % 2]
        y_true = predictions[output_name]['test_true']
        y_pred = predictions[output_name]['test_pred']

        ax.scatter(y_true, y_pred, alpha=0.7,
                   edgecolors='black', linewidth=0.5)

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


def plot_membership_functions(system, output_name='Stability',
                              output_path='output/membership_functions_stability.png'):
    """
    Visualizes Gaussian membership functions for each input variable.
    Purpose: Shows how fuzzy sets are defined after subtractive clustering and tuning.
    Note: Only generated for Stability as a representative example.
    """
    n_rules = system.n_rules()
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


def generate_all_plots(histories, predictions, results, systems):
    print("\nGenerating plots...")

    plot_training_convergence(histories)
    plot_predicted_vs_actual_test(predictions)
    plot_membership_functions(systems['Stability'], output_name='Stability')

    print("All plots saved to output/")
