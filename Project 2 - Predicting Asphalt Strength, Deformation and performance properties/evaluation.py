"""
Evaluation metrics for the TSK Fuzzy System.
"""

import numpy as np


def rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
