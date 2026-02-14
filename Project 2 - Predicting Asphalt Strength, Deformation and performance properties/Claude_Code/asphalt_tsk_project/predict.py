import os
import sys
import pickle
import numpy as np

from config import INPUT_COLUMNS, OUTPUT_COLUMNS


def load_model(model_path: str = "output/trained_systems.pkl"):
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at '{model_path}'.")
        print("Please run main.py first to train the system.")
        sys.exit(1)

    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    return bundle["systems"], bundle["normaliser"]


def predict_single(systems, normaliser, raw_input: np.ndarray) -> dict:
    """
    Make a prediction for a single input vector.

    Parameters
    ----------
    raw_input : (10,) array of raw (un-normalised) input values

    Returns
    -------
    predictions : dict mapping output name -> predicted value
    """
    X = raw_input.reshape(1, -1)
    X_norm = normaliser.transform_X(X)

    predictions = {}
    for idx, name in enumerate(OUTPUT_COLUMNS):
        y_norm = systems[name].predict(X_norm)
        y_orig = normaliser.output_scalers[name].inverse_transform(
            y_norm.reshape(-1, 1)
        ).ravel()[0]
        predictions[name] = y_orig

    return predictions


def interactive_mode():
    """Run an interactive prompt loop for end-user predictions."""
    systems, normaliser = load_model()

    print("=" * 55)
    print("  Asphalt Properties Prediction System (TSK Fuzzy)")
    print("=" * 55)
    print()

    while True:
        print("Enter the 10 input parameters (or 'q' to quit):\n")
        values = []
        for col_name in INPUT_COLUMNS:
            while True:
                raw = input(f"  {col_name}: ").strip()
                if raw.lower() == "q":
                    print("Exiting.")
                    return
                try:
                    values.append(float(raw))
                    break
                except ValueError:
                    print("    Invalid number. Please try again.")

        raw_input = np.array(values)
        preds = predict_single(systems, normaliser, raw_input)

        print("\n   Predicted Outputs ")
        for name, val in preds.items():
            print(f"  {name:>12}: {val:.4f}")
        print()


if __name__ == "__main__":
    interactive_mode()
