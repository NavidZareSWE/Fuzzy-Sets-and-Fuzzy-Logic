# Fuzzy Time Series Forecasting - Comprehensive Report

**Generated:** 2026-01-07 22:22:56

## Executive Summary

This report presents the results of applying Fuzzy Time Series (FTS) forecasting to multiple datasets using both First-Order FTS (FOFTS) and High-Order FTS (HOFTS) with various membership function types.

### Best Configurations by Dataset

| Dataset | Order | Partitions | MF Type | Test RMSE | Test MAE | Test MAPE (%) |
|---------|-------|------------|---------|-----------|----------|---------------|
| Mackey-Glass | 4 | 17 | trapezoidal | 0.046549 | 0.036178 | 6.20 |
| Total Specimens | 1 | 9 | triangular | 9284.725846 | 7695.288542 | 11.16 |
| Influenza A | 2 | 9 | trapezoidal | 2975.629976 | 1985.286111 | 299.65 |
| Influenza B | 1 | 11 | trapezoidal | 371.375746 | 273.059091 | 101.36 |

---

## Dataset: Mackey-Glass

### Data Summary

- **Training samples:** 800
- **Testing samples:** 200
- **Total samples:** 1000
- **Data range:** [0.0359, 1.5924]

### Best Model Configuration

- **Order:** 4 (HOFTS (order 4))
- **Number of Partitions:** 17
- **Membership Function:** Trapezoidal

### Performance Metrics (Best Model)

| Metric | Training | Testing |
|--------|----------|----------|
| RMSE | 0.040316 | 0.046549 |
| MAE | 0.032475 | 0.036178 |
| MAPE (%) | 7.73 | 6.20 |

### First-Order vs High-Order Comparison

|   Order | Type            |   Partitions | MF Type     |   Test RMSE |   Test MAE |   Test MAPE |
|--------:|:----------------|-------------:|:------------|------------:|-----------:|------------:|
|       1 | FOFTS           |           15 | triangular  |   0.0608091 |  0.0507805 |    10.3166  |
|       2 | HOFTS (order=2) |           17 | triangular  |   0.0569616 |  0.0466467 |     8.55496 |
|       3 | HOFTS (order=3) |           17 | trapezoidal |   0.0504973 |  0.0400919 |     6.53036 |
|       4 | HOFTS (order=4) |           17 | trapezoidal |   0.0465489 |  0.0361783 |     6.20019 |
|       5 | HOFTS (order=5) |           17 | trapezoidal |   0.056383  |  0.0410596 |     7.21408 |

### Sample FLRGs (Best Model)

```
(A6, A6, A5, A5) -> A5, A6, A4
(A6, A5, A5, A5) -> A4, A5, A6
(A5, A5, A5, A4) -> A4
(A5, A5, A4, A4) -> A4
(A5, A4, A4, A4) -> A4, A5
(A4, A4, A4, A4) -> A3, A4, A5
(A4, A4, A4, A3) -> A3
(A4, A4, A3, A3) -> A3, A4
(A4, A3, A3, A3) -> A3
(A3, A3, A3, A3) -> A3, A2, A4
(A3, A3, A3, A2) -> A2
(A3, A3, A2, A2) -> A2, A3
(A3, A2, A2, A2) -> A2
(A2, A2, A2, A2) -> A2, A3
(A2, A2, A2, A3) -> A4
(A2, A2, A3, A4) -> A4, A5
(A2, A3, A4, A4) -> A5
(A3, A4, A4, A5) -> A5
(A4, A4, A5, A5) -> A5, A6
(A4, A5, A5, A5) -> A5
... and 153 more FLRGs
```

---

## Dataset: Total Specimens

### Data Summary

- **Training samples:** 190
- **Testing samples:** 48
- **Total samples:** 238
- **Data range:** [22366.0000, 191785.0000]

### Best Model Configuration

- **Order:** 1 (FOFTS)
- **Number of Partitions:** 9
- **Membership Function:** Triangular

### Performance Metrics (Best Model)

| Metric | Training | Testing |
|--------|----------|----------|
| RMSE | 11496.012397 | 9284.725846 |
| MAE | 9517.017989 | 7695.288542 |
| MAPE (%) | 18.08 | 11.16 |

### First-Order vs High-Order Comparison

|   Order | Type            |   Partitions | MF Type     |   Test RMSE |   Test MAE |   Test MAPE |
|--------:|:----------------|-------------:|:------------|------------:|-----------:|------------:|
|       1 | FOFTS           |            9 | triangular  |     9284.73 |    7695.29 |    11.1595  |
|       2 | HOFTS (order=2) |            9 | triangular  |     9718.27 |    8350.56 |    11.9196  |
|       3 | HOFTS (order=3) |           13 | triangular  |     9730.14 |    7524.06 |     9.2432  |
|       4 | HOFTS (order=4) |           17 | trapezoidal |    11122    |    7613.13 |     8.23235 |
|       5 | HOFTS (order=5) |           17 | trapezoidal |    12127.3  |    8298.34 |     9.31059 |

### Sample FLRGs (Best Model)

```
A3 -> A2, A3, A4
A2 -> A2, A3
A4 -> A4, A5, A3
A5 -> A5, A6, A4
A6 -> A7, A5, A6
A7 -> A7, A6, A8
A8 -> A8, A7
```

---

## Dataset: Influenza A

### Data Summary

- **Training samples:** 190
- **Testing samples:** 48
- **Total samples:** 238
- **Data range:** [3.0000, 53136.0000]

### Best Model Configuration

- **Order:** 2 (HOFTS (order 2))
- **Number of Partitions:** 9
- **Membership Function:** Trapezoidal

### Performance Metrics (Best Model)

| Metric | Training | Testing |
|--------|----------|----------|
| RMSE | 1859.069076 | 2975.629976 |
| MAE | 1601.047872 | 1985.286111 |
| MAPE (%) | 3186.04 | 299.65 |

### First-Order vs High-Order Comparison

|   Order | Type            |   Partitions | MF Type     |   Test RMSE |   Test MAE |   Test MAPE |
|--------:|:----------------|-------------:|:------------|------------:|-----------:|------------:|
|       1 | FOFTS           |           13 | triangular  |     4095.71 |    3128.33 |     466.601 |
|       2 | HOFTS (order=2) |            9 | trapezoidal |     2975.63 |    1985.29 |     299.65  |
|       3 | HOFTS (order=3) |            9 | trapezoidal |     3062.82 |    2078.86 |     304.839 |
|       4 | HOFTS (order=4) |            9 | trapezoidal |     3811.96 |    2207.3  |     306.479 |
|       5 | HOFTS (order=5) |           13 | triangular  |     3953.64 |    2778.13 |     462.013 |

### Sample FLRGs (Best Model)

```
(A1, A1) -> A1, A2
(A1, A2) -> A2
(A2, A2) -> A2, A1, A3
(A2, A1) -> A2, A1
(A2, A3) -> A4
(A3, A4) -> A5
(A4, A5) -> A6, A5
(A5, A6) -> A7
(A6, A7) -> A9
(A7, A9) -> A9
(A9, A9) -> A8
(A9, A8) -> A6
(A8, A6) -> A4
(A6, A4) -> A3
(A4, A3) -> A2, A3
(A3, A2) -> A2
(A5, A5) -> A4
(A5, A4) -> A3
(A3, A3) -> A3, A2
```

---

## Dataset: Influenza B

### Data Summary

- **Training samples:** 190
- **Testing samples:** 48
- **Total samples:** 238
- **Data range:** [9.0000, 8185.0000]

### Best Model Configuration

- **Order:** 1 (FOFTS)
- **Number of Partitions:** 11
- **Membership Function:** Trapezoidal

### Performance Metrics (Best Model)

| Metric | Training | Testing |
|--------|----------|----------|
| RMSE | 372.185186 | 371.375746 |
| MAE | 232.372294 | 273.059091 |
| MAPE (%) | 206.60 | 101.36 |

### First-Order vs High-Order Comparison

|   Order | Type            |   Partitions | MF Type     |   Test RMSE |   Test MAE |   Test MAPE |
|--------:|:----------------|-------------:|:------------|------------:|-----------:|------------:|
|       1 | FOFTS           |           11 | trapezoidal |     371.376 |    273.059 |     101.364 |
|       2 | HOFTS (order=2) |            7 | trapezoidal |     595.261 |    508.642 |     382.453 |
|       3 | HOFTS (order=3) |           17 | trapezoidal |     630.745 |    453.392 |     207.904 |
|       4 | HOFTS (order=4) |            9 | trapezoidal |     673.507 |    454.997 |     179.729 |
|       5 | HOFTS (order=5) |            9 | trapezoidal |     686.419 |    470.133 |     185.1   |

### Sample FLRGs (Best Model)

```
A2 -> A1, A2, A3
A1 -> A1, A2
A3 -> A3, A4, A2
A4 -> A5, A3
A5 -> A7, A4
A7 -> A7, A6, A9
A6 -> A6, A7, A5
A9 -> A10, A8
A10 -> A10, A11, A9
A11 -> A10
A8 -> A7
```

---

## Methodology

### Fuzzy Set Partitioning

The universe of discourse is partitioned into fuzzy sets using the following membership function types:

1. **Triangular:** Classic triangular membership functions with 50% overlap
2. **Trapezoidal:** Trapezoidal functions with flat top regions
3. **Gaussian:** Bell-shaped Gaussian curves
4. **Generalized Bell:** Adjustable bell-shaped functions

### FLRG Generation

For a model of order k, FLRGs are generated as:
- **First-Order (k=1):** F(t-1) → F(t)
- **High-Order (k≥2):** (F(t-k), F(t-k+1), ..., F(t-1)) → F(t)

### Defuzzification

The centroid method is used for defuzzification, computing the average center of all consequent fuzzy sets in the matched FLRG.

### Performance Metrics

- **RMSE:** Root Mean Square Error = √(mean((actual - predicted)²))
- **MAE:** Mean Absolute Error = mean(|actual - predicted|)
- **MAPE:** Mean Absolute Percentage Error = mean(|actual - predicted| / |actual|) × 100

---

## Visualizations

The following visualizations are available in the results directory:

- Membership function plots for each dataset
- Actual vs Predicted time series plots
- Error metric heatmaps across parameter space
- Fuzzification visualization

