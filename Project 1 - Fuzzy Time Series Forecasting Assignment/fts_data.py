import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path


class DataLoader:
    """
    Data loader for time series datasets.
    Supports CSV and Excel file formats.
    """

    def load_mackey_glass(self, filepath: str) -> np.ndarray:
        """
        Load the Mackey-Glass time series dataset.

        Args:
            filepath: Path to the CSV file

        Returns:
            1D numpy array of time series values
        """
        df = pd.read_csv(filepath)

        # The file has 'index' and 'value' columns
        if 'value' in df.columns:
            data = df['value'].values
        else:
            # Assume second column is the value
            data = df.iloc[:, 1].values

        return data.astype(np.float64)

    def load_influenza_data(self, filepath: str) -> Dict[str, np.ndarray]:
        """
        Load the Influenza specimens dataset.

        Args:
            filepath: Path to the Excel file

        Returns:
            Dictionary with keys:
            - 'total_specimens': Total number of specimens
            - 'influenza_a': Cases of Influenza A
            - 'influenza_b': Cases of Influenza B
            - 'percent_positive': Percent positive
            - 'metadata': DataFrame with year/week info
        """
        # Read Excel file with header in second row
        df = pd.read_excel(filepath, header=1)

        # Clean column names
        df.columns = [col.strip() if isinstance(
            col, str) else col for col in df.columns]

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

    def preprocess(self, data: np.ndarray,
                   normalize: bool = False,
                   remove_nan: bool = True,
                   smooth: bool = False,
                   window_size: int = 3) -> np.ndarray:
        """
        Preprocess time series data.

        Args:
            data: Raw time series data
            normalize: Whether to normalize to [0, 1] range
            remove_nan: Whether to remove NaN values
            smooth: Whether to apply moving average smoothing
            window_size: Window size for smoothing

        Returns:
            Preprocessed data
        """
        result = data.copy()

        # Remove NaN values
        if remove_nan:
            result = result[~np.isnan(result)]

        # Apply smoothing
        if smooth and len(result) > window_size:
            kernel = np.ones(window_size) / window_size
            result = np.convolve(result, kernel, mode='valid')

        # Normalize
        if normalize:
            min_val = np.min(result)
            max_val = np.max(result)
            if max_val > min_val:
                result = (result - min_val) / (max_val - min_val)

        return result

    def train_test_split(self, data: np.ndarray,
                         train_ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split data into training and testing sets.

        Args:
            data: Time series data
            train_ratio: Proportion of data for training

        Returns:
            Tuple of (train_data, test_data)
        """
        train_size = int(len(data) * train_ratio)
        return data[:train_size], data[train_size:]

    def create_sliding_window(self, data: np.ndarray,
                              window_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding window samples for machine learning.

        Args:
            data: Time series data
            window_size: Number of past values to use as features

        Returns:
            Tuple of (X, y) where X has shape (n_samples, window_size)
            and y has shape (n_samples,)
        """
        n_samples = len(data) - window_size
        X = np.zeros((n_samples, window_size))
        y = np.zeros(n_samples)

        for i in range(n_samples):
            X[i] = data[i:i + window_size]
            y[i] = data[i + window_size]

        return X, y


def load_datasets(mackey_glass_path: str,
                  influenza_path: str) -> Dict[str, np.ndarray]:
    """
    Load all datasets from the provided file paths.

    Args:
        mackey_glass_path: Path to Mackey-Glass CSV file
        influenza_path: Path to Influenza Excel file

    Returns:
        Dictionary containing all time series data
    """
    datasets = {}

    # Create DataLoader instance
    loader = DataLoader()

    # Load Mackey-Glass
    datasets['Mackey-Glass'] = loader.load_mackey_glass(mackey_glass_path)

    # Load Influenza data
    influenza_data = loader.load_influenza_data(influenza_path)
    datasets['Total Specimens'] = influenza_data['total_specimens']
    datasets['Influenza A'] = influenza_data['influenza_a']
    datasets['Influenza B'] = influenza_data['influenza_b']

    return datasets


def describe_dataset(data: np.ndarray, name: str = "Dataset"):
    """
    Print descriptive statistics for a dataset.

    Args:
        data: Time series data
        name: Name of the dataset
    """
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
