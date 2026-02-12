import numpy as np
import pandas as pd
from pathlib import Path


class DataLoader:
    def load_mackey_glass(self, filepath):
        df = pd.read_csv(filepath)

        if 'value' in df.columns:
            data = df['value'].values
        else:
            data = df.iloc[:, 1].values

        return data.astype(np.float64)

    def load_influenza_data(self, filepath):
        df = pd.read_excel(filepath, header=1)

        # Clean column names
        # Clean the DataFrame column names by removing any leading or trailing whitespace.
        df.columns = [
            col.strip() if isinstance(col, str) else col
            for col in df.columns
        ]

        result = {
            'total_specimens': df['TOTAL SPECIMENS'].values.astype(np.float64),
            'influenza_a': df['TOTAL A'].values.astype(np.float64),
            'influenza_b': df['TOTAL B'].values.astype(np.float64),
            'percent_positive': df['PERCENT POSITIVE'].values.astype(np.float64),
            'percent_a': df['PERCENT A'].values.astype(np.float64),
            'percent_b': df['PERCENT B'].values.astype(np.float64),
            'metadata': df[['YEAR', 'WEEK']]
        }

        return result

    def preprocess(self, data, should_normalize=False, should_remove_nan=True, should_smooth=False, window_size=3):
        result = data.copy()

        if should_remove_nan:
            result = result[~np.isnan(result)]

        # Smoothing enhances data visualization by reducing noise, allowing clearer insights into patterns or trends.
        # This code implements a simple moving average smoothing technique using convolution with a uniform kernel.
        if should_smooth and len(result) > window_size:
            kernel = np.ones(window_size) / window_size
            result = np.convolve(result, kernel, mode='valid')

        if should_normalize:
            min_val = np.min(result)
            max_val = np.max(result)
            if max_val > min_val:
                result = (result - min_val) / (max_val - min_val)

        return result

    def train_test_split(self, data, train_ratio=0.8):
        train_size = int(len(data) * train_ratio)
        return data[:train_size], data[train_size:]


def load_datasets(mackey_glass_path, influenza_path):
    datasets = {}
    loader = DataLoader()

    datasets['Mackey-Glass'] = loader.load_mackey_glass(mackey_glass_path)

    influenza_data = loader.load_influenza_data(influenza_path)
    datasets['Total Specimens'] = influenza_data['total_specimens']
    datasets['Influenza A'] = influenza_data['influenza_a']
    datasets['Influenza B'] = influenza_data['influenza_b']

    return datasets


def describe_dataset(data, name="Dataset"):
    print(f"\n{'='*50}")
    print(f"Dataset: {name}")
    print(f"{'='*50}")
    print(f"  Length: {len(data)}")
    print(f"  Min: {np.min(data):.4f}")
    print(f"  Max: {np.max(data):.4f}")
    print(f"  Mean: {np.mean(data):.4f}")
    print(f"  Std: {np.std(data):.4f}")
    print(f"  Median: {np.median(data):.4f}")

    # Check for NaN values
    nan_count = np.sum(np.isnan(data))
    if nan_count > 0:
        print(f"  NaN values: {nan_count}")
