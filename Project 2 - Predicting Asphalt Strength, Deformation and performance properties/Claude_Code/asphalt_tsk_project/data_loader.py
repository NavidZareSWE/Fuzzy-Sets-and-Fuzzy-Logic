"""
Data loading, preprocessing, and train/test splitting utilities.

The Excel file has two header rows (row 0 = group labels, row 1 = column names)
and one description row (row 2). Actual numeric data begins at row 3.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from config import (
    DATA_PATH, INPUT_COLUMNS, OUTPUT_COLUMNS, ALL_COLUMNS,
    TEST_RATIO, RANDOM_SEED,
)


def load_raw_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Read the Excel file and return a clean numeric DataFrame."""
    raw = pd.read_excel(path, header=None)
    # Data rows start at index 3 (0-indexed); first three rows are headers/descriptions
    data = raw.iloc[3:].reset_index(drop=True)
    data.columns = ALL_COLUMNS
    data = data.apply(pd.to_numeric, errors="coerce")
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data


def split_data(df: pd.DataFrame):
    """
    Split the dataset into 80 % train and 20 % test (stratification is not
    applied because the outputs are continuous).

    Returns
    -------
    X_train, X_test : np.ndarray   — input matrices
    y_train, y_test : np.ndarray   — output matrices (columns in OUTPUT_COLUMNS order)
    """
    X = df[INPUT_COLUMNS].values.astype(np.float64)
    Y = df[OUTPUT_COLUMNS].values.astype(np.float64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=TEST_RATIO, random_state=RANDOM_SEED
    )
    return X_train, X_test, y_train, y_test


class DataNormaliser:
    """
    Min-max normalisation to [0, 1].  Fitted on training data only, then
    applied to both training and test data to prevent information leakage.
    """

    def __init__(self):
        self.input_scaler = MinMaxScaler()
        self.output_scalers = {}  # one per output column

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.input_scaler.fit(X_train)
        for i, col_name in enumerate(OUTPUT_COLUMNS):
            scaler = MinMaxScaler()
            scaler.fit(y_train[:, i].reshape(-1, 1))
            self.output_scalers[col_name] = scaler

    def transform_X(self, X: np.ndarray) -> np.ndarray:
        return self.input_scaler.transform(X)

    def inverse_transform_X(self, X_norm: np.ndarray) -> np.ndarray:
        return self.input_scaler.inverse_transform(X_norm)

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        y_norm = np.zeros_like(y)
        for i, col_name in enumerate(OUTPUT_COLUMNS):
            y_norm[:, i] = self.output_scalers[col_name].transform(
                y[:, i].reshape(-1, 1)
            ).ravel()
        return y_norm

    def inverse_transform_y(self, y_norm: np.ndarray) -> np.ndarray:
        y_orig = np.zeros_like(y_norm)
        for i, col_name in enumerate(OUTPUT_COLUMNS):
            y_orig[:, i] = self.output_scalers[col_name].inverse_transform(
                y_norm[:, i].reshape(-1, 1)
            ).ravel()
        return y_orig


if __name__ == "__main__":
    df = load_raw_data()
    print(f"Loaded {len(df)} samples with columns: {list(df.columns)}")
    X_tr, X_te, y_tr, y_te = split_data(df)
    print(f"Train: {X_tr.shape[0]},  Test: {X_te.shape[0]}")
