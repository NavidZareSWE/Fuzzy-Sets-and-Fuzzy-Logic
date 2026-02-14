import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
        Root Mean Square Error:

        RMSE = sqrt(mean((actual - predicted)^2))
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def print_results_table(results: dict):
    header = f"{'Output':<12} {'RMSE(Train)':>14} {'RMSE(Test)':>14}"
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for name, (tr, te) in results.items():
        print(f"{name:<12} {tr:>14.4f} {te:>14.4f}")
    print("=" * len(header))
