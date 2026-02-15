import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from config import (
    DATA_PATH,
    INPUT_COLUMNS,
    OUTPUT_COLUMNS,
    ALL_COLUMNS,
    TEST_RATIO,
    RANDOM_SEED,
)


def load_raw_data(path=DATA_PATH):
    # print("Resolved DATA_PATH:", path)
    raw = pd.read_excel(path, header=None)
    # First three rows are headers
    #   Selects rows from the fourth row onward,
    #   resets the index,
    #   and assigns the modified DataFrame to 'data'.
    data = raw.iloc[3:].reset_index(drop=True)
    data.columns = ALL_COLUMNS
    # Converts all values in the 'data' DataFrame to numeric, replacing non-numeric entries with NaN to prevent errors in subsequent analyses.
    data = data.apply(pd.to_numeric, errors="coerce")
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data


def split_data(df):
    X = df[INPUT_COLUMNS].values.astype(np.float64)
    Y = df[OUTPUT_COLUMNS].values.astype(np.float64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=TEST_RATIO, random_state=RANDOM_SEED
    )
    return X_train, X_test, y_train, y_test


class DataNormaliser:
    """Min-max normalisation to [0, 1]."""

    def __init__(self):
        self.input_scaler = MinMaxScaler()
        self.output_scalers = {}

    def _fit(self, X_train, y_train):
        self.input_scaler.fit(X_train)
        for i, col_name in enumerate(OUTPUT_COLUMNS):
            scaler = MinMaxScaler()
            scaler.fit(y_train[:, i].reshape(-1, 1))
            self.output_scalers[col_name] = scaler

    def _transform_X(self, X):
        return self.input_scaler.transform(X)

    def _inverse_transform_X(self, X_norm):
        return self.input_scaler.inverse_transform(X_norm)

    def _transform_y(self, y):
        y_norm = np.zeros_like(y)
        for i, col_name in enumerate(OUTPUT_COLUMNS):
            y_norm[:, i] = self.output_scalers[col_name].transform(
                y[:, i].reshape(-1, 1)
            ).ravel()
        return y_norm

    def _inverse_transform_y(self, y_norm):
        y_orig = np.zeros_like(y_norm)
        for i, col_name in enumerate(OUTPUT_COLUMNS):
            y_orig[:, i] = self.output_scalers[col_name].inverse_transform(
                y_norm[:, i].reshape(-1, 1)
            ).ravel()
        return y_orig
