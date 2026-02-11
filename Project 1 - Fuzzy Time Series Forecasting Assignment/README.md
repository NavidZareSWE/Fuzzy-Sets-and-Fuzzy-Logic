# Fuzzy Time Series Forecasting Project

A complete implementation of First-Order and High-Order Fuzzy Time Series (FTS) forecasting models from scratch, without using any pre-implemented fuzzy time-series libraries.

## Project Overview

This project implements Fuzzy Time Series forecasting methods for:
1. **Mackey-Glass Time Series** - A chaotic, non-linear benchmark dataset
2. **Influenza Specimens Data** - Real-world epidemiological time series

### Features

- **First-Order FTS (FOFTS)**: Uses single previous value for prediction
- **High-Order FTS (HOFTS)**: Uses multiple previous values (order >= 2)
- **Multiple Membership Functions**: Triangular, Trapezoidal, Gaussian, Bell-shaped
- **Automatic Parameter Tuning**: Grid search over order and partition space
- **Performance Metrics**: RMSE, MAE, MAPE
- **Interactive Prediction Interface**: CLI for making predictions
- **Comprehensive Visualizations**: Membership functions, predictions, error analysis

## Project Structure

```
fuzzy_time_series/
├── __init__.py           # Package initialization
├── fts_core.py           # Core FTS algorithms and data structures
├── fts_data.py           # Data loading and preprocessing
├── fts_experiments.py    # Experiment runner and parameter tuning
├── fts_visualization.py  # Plotting and visualization tools
├── fts_interface.py      # User interface for predictions
├── fts_report.py         # Report generation utilities
├── main.py               # Main execution script
├── run_interactive.py    # Interactive prediction interface
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Installation

```bash
# Install required dependencies
pip install -r requirements.txt
```

## Usage

### Running Complete Analysis

To run the full analysis on all datasets:

```bash
python main.py
```

This will:
1. Load both datasets
2. Run systematic experiments with various configurations
3. Generate comprehensive reports and visualizations
4. Save results to the `results/` directory

### Interactive Prediction Interface

To use the interactive prediction interface:

```bash
python run_interactive.py
# Or with custom parameters:
python run_interactive.py --dataset mackey_glass --order 3 --partitions 11 --mf-type triangular
```

Available commands in the interface:
- `predict` (p): Enter values and get single prediction
- `multi` (m): Forecast multiple steps ahead
- `flrgs` (f): Display all Fuzzy Logical Relation Groups
- `info` (i): Show model information
- `help` (h): Show help message
- `quit` (q): Exit the program

### Using as a Library

```python
from fts_core import FuzzyTimeSeries, MembershipFunctionType
from fts_data import DataLoader
import numpy as np

# Load data
data = DataLoader.load_mackey_glass('path/to/mackey_glass.csv')

# Split data
train_data = data[:int(len(data) * 0.8)]

# Create and train model
model = FuzzyTimeSeries(
    order=3,                                      # High-order FTS
    num_partitions=11,                            # Number of fuzzy sets
    mf_type=MembershipFunctionType.TRIANGULAR    # Membership function type
)
model.fit(train_data)

# Make predictions
history = list(train_data[-10:])
prediction = model.predict_next(history)
print(f"Predicted next value: {prediction}")

# Multi-step forecast
forecasts = model.forecast(steps=5, initial_history=history)
print(f"5-step forecast: {forecasts}")
```

## Methodology

### 1. Universe of Discourse Definition

The universe of discourse is defined as:
- U = [D_min - margin, D_max + margin]
- Where margin = 10% of data range

### 2. Fuzzy Set Partitioning

The universe is partitioned into n fuzzy sets using:

**Triangular MF**: Classic triangular functions with 50% overlap
```
μ(x) = max(0, min((x-a)/(b-a), (c-x)/(c-b)))
```

**Trapezoidal MF**: Flat plateau regions for robust classification
```
μ(x) = max(0, min((x-a)/(b-a), 1, (d-x)/(d-c)))
```

**Gaussian MF**: Smooth bell curves
```
μ(x) = exp(-((x-c)²)/(2σ²))
```

**Generalized Bell MF**: Adjustable shape
```
μ(x) = 1 / (1 + |((x-c)/a)|^(2b))
```

### 3. FLRG Generation

**First-Order (k=1)**:
```
F(t-1) -> F(t)
```

**High-Order (k>=2)**:
```
(F(t-k), F(t-k+1), ..., F(t-1)) -> F(t)
```

### 4. Defuzzification

Centroid method: average of consequent fuzzy set centers.

## Performance Metrics

- **RMSE**: Root Mean Square Error = [ok](mean((actual - predicted)²))
- **MAE**: Mean Absolute Error = mean(|actual - predicted|)
- **MAPE**: Mean Absolute Percentage Error = mean(|actual - predicted|/|actual|) × 100

## Output Files

After running `main.py`, the following outputs are generated:

```
results/
├── mackey_glass/
│   ├── results.csv                    # All experiment results
│   ├── summary.json                   # Best configuration summary
│   ├── best_membership_functions.png  # Membership function plot
│   ├── best_predictions.png           # Prediction vs actual plot
│   └── heatmap_*.png                  # Parameter heatmaps
├── total_specimens/
│   └── ...
├── influenza_a/
│   └── ...
├── influenza_b/
│   └── ...
├── appendices/
│   ├── *_flrgs.txt                    # Complete FLRG listings
│   ├── *_fuzzy_sets.txt               # Fuzzy set definitions
│   └── *_best_config.json             # Best configurations
├── final_summary.csv                  # Cross-dataset summary
├── COMPREHENSIVE_REPORT.md            # Markdown report
└── FULL_REPORT.txt                    # Text report
```

## Example Results

### Mackey-Glass Dataset
- Best Configuration: Order=3, Partitions=11, MF=Triangular
- Test RMSE: ~0.05

### Influenza Data
- Best configurations vary by target variable
- Generally benefits from higher-order models due to seasonal patterns

## References

1. Song, Q., & Chissom, B. S. (1993). Fuzzy time series and its models.
2. Chen, S. M. (1996). Forecasting enrollments based on fuzzy time series.
3. Hwang, J. R., Chen, S. M., & Lee, C. H. (1998). Handling forecasting problems using fuzzy time series.

## Note

This implementation is created from scratch without using any pre-implemented fuzzy time-series libraries, as per the project requirements. All fuzzy logic algorithms, membership functions, and FTS methods are implemented manually.
