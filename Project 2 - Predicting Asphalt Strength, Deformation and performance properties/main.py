import os
import numpy as np

from config import OUTPUT_COLUMNS, RANDOM_SEED, CLUSTER_RADIUS
from data_loader import load_raw_data, split_data, DataNormaliser
from training import build_tsk_system, tune_tsk_system
from visualization import generate_all_plots
from results_writer import save_all_results


def rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def train_systems(X_train, y_train):
    """Train one TSK system per output variable."""
    systems = {}
    histories = {}

    for idx, output_name in enumerate(OUTPUT_COLUMNS):
        y_tr_col = y_train[:, idx]
        system = build_tsk_system(
            X_train, y_tr_col, cluster_radius=CLUSTER_RADIUS)
        history = tune_tsk_system(system, X_train, y_tr_col)
        systems[output_name] = system
        histories[output_name] = history

    return systems, histories


def evaluate_systems(systems, normaliser, X_train, X_test, y_train_raw, y_test_raw):
    results = {}
    predictions = {}

    for idx, output_name in enumerate(OUTPUT_COLUMNS):
        system = systems[output_name]

        y_pred_train_norm = system.predict(X_train)
        y_pred_test_norm = system.predict(X_test)

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

    return results, predictions


def print_results(results, systems):
    print("\n" + "=" * 50)
    print("RMSE RESULTS")
    print("=" * 50)
    print(f"{'Output':<12} {'RMSE(Train)':>14} {'RMSE(Test)':>14}")
    print("-" * 50)
    for name, (tr, te) in results.items():
        print(f"{name:<12} {tr:>14.4f} {te:>14.4f}")
    print("=" * 50)

    print("\n" + "=" * 50)
    print("NUMBER OF RULES PER OUTPUT")
    print("=" * 50)
    for name, system in systems.items():
        print(f"{name:<12}: {system.n_rules()} rules")
    print("=" * 50)


def main():
    np.random.seed(RANDOM_SEED)
    os.makedirs("output", exist_ok=True)

    # 1. Load and split data
    print("Loading and splitting data...")
    df = load_raw_data()
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = split_data(df)

    # 2. Normalise data
    print("Normalising data...")
    normaliser = DataNormaliser()
    normaliser._fit(X_train_raw, y_train_raw)

    X_train = normaliser._transform_X(X_train_raw)
    X_test = normaliser._transform_X(X_test_raw)
    y_train = normaliser._transform_y(y_train_raw)
    y_test = normaliser._transform_y(y_test_raw)

    # 3. Train systems
    print("Training TSK systems...")
    systems, histories = train_systems(X_train, y_train)

    # 4. Evaluate systems
    print("Evaluating systems...")
    results, predictions = evaluate_systems(
        systems, normaliser, X_train, X_test, y_train_raw, y_test_raw
    )

    # 5. Print results to console
    print_results(results, systems)

    # 6. Generate all plots
    generate_all_plots(histories, predictions, results, systems)

    # 7. Save all results
    # Set save_model=True if you need to reuse the trained model later
    save_all_results(
        results, systems, normaliser, df, X_train_raw, X_test_raw,
        save_model=False
    )

    return results


if __name__ == "__main__":
    results = main()
