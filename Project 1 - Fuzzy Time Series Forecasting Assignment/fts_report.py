import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np


def generate_flrg_appendix(model, output_path: str):
    """Generate a text file containing all FLRGs for a model."""
    with open(output_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("FUZZY LOGICAL RELATION GROUPS (FLRGs)\n")
        f.write("="*60 + "\n\n")

        f.write(f"Model Configuration:\n")
        f.write(f"  Order: {model.order}\n")
        f.write(f"  Number of Partitions: {model.num_partitions}\n")
        f.write(f"  Membership Function: {model.mf_type.value}\n")
        f.write(f"  Total FLRGs: {len(model.flrgs)}\n\n")

        f.write("-"*60 + "\n")
        f.write("FLRGs:\n")
        f.write("-"*60 + "\n\n")

        for i, flrg in enumerate(model.flrgs.values(), 1):
            f.write(f"{i:4d}. {flrg}\n")


def generate_fuzzy_sets_appendix(model, output_path: str):
    """Generate a text file containing fuzzy set definitions."""
    with open(output_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("FUZZY SET DEFINITIONS\n")
        f.write("="*60 + "\n\n")

        f.write(f"Universe of Discourse:\n")
        f.write(f"  Lower Bound: {model.universe.lower_bound:.6f}\n")
        f.write(f"  Upper Bound: {model.universe.upper_bound:.6f}\n")
        f.write(f"  Range: {model.universe.range:.6f}\n\n")

        f.write(
            f"Membership Function Type: {model.mf_type.value.capitalize()}\n")
        f.write(f"Number of Partitions: {model.num_partitions}\n\n")

        f.write("-"*60 + "\n")
        f.write("Fuzzy Sets:\n")
        f.write("-"*60 + "\n\n")

        for fs in model.fuzzy_sets:
            f.write(f"Fuzzy Set: {fs.name}\n")
            f.write(f"  Center: {fs.center:.6f}\n")
            f.write(f"  Parameters:\n")
            for param, value in fs.parameters.items():
                f.write(f"    {param}: {value:.6f}\n")
            f.write("\n")


def generate_methodology_section() -> str:
    """Generate the methodology section text."""
    return """
## Methodology

### 1. Universe of Discourse Definition

The universe of discourse U is defined based on the range of training data values:
- U = [D_min - margin, D_max + margin]
- Where margin = (D_max - D_min) × 0.1 (10% of data range)

This margin ensures that test data values near the boundaries can be properly fuzzified.

### 2. Fuzzy Set Partitioning Strategy

The universe of discourse is partitioned into n equal-width fuzzy sets. We experimented with the following membership function types:

#### 2.1 Triangular Membership Functions
The triangular MF is defined by three parameters (a, b, c):
- μ(x) = 0 if x ≤ a or x >= c
- μ(x) = (x - a)/(b - a) if a < x ≤ b
- μ(x) = (c - x)/(c - b) if b < x < c

Adjacent fuzzy sets overlap at the 0.5 membership level, ensuring complete coverage.

#### 2.2 Trapezoidal Membership Functions
The trapezoidal MF is defined by four parameters (a, b, c, d):
- Provides flat plateau regions for more robust classification
- First and last sets are half-trapezoids extending beyond the universe

#### 2.3 Gaussian Membership Functions
The Gaussian MF is defined by center c and standard deviation σ:
- μ(x) = exp(-((x - c)² / (2σ²)))
- σ is chosen so adjacent Gaussians cross at ~0.5 membership

#### 2.4 Generalized Bell Membership Functions
The bell MF offers adjustable shape via parameters (a, b, c):
- μ(x) = 1 / (1 + |((x - c) / a)|^(2b))
- Parameter b controls slope steepness

### 3. Membership Function Creation Process

1. Calculate universe bounds with margin
2. Determine partition width: width = (upper - lower) / (n - 1) for n partitions
3. Calculate center points evenly spaced across universe
4. Define membership function parameters based on type:
   - For triangular: left = center - width, right = center + width
   - For Gaussian: σ = width / (2 × [ok](2 × ln(2)))

### 4. Fuzzification

Each crisp value x is converted to a fuzzy linguistic variable by:
1. Computing membership degrees for all fuzzy sets
2. Selecting the fuzzy set with maximum membership
3. Creating the fuzzified time series F(t)

### 5. FLRG Generation Process

#### First-Order FTS (FOFTS)
For each consecutive pair (F(t), F(t+1)), create relation:
- F(t) -> F(t+1)

Group relations with same antecedent:
- A1 -> A2, A3, A4 means "when state is A1, next states could be A2, A3, or A4"

#### High-Order FTS (HOFTS)
For order k, create relations using k previous values:
- (F(t-k+1), F(t-k+2), ..., F(t)) -> F(t+1)

This captures longer temporal dependencies and complex patterns.

### 6. Forecasting Algorithm

Given historical values, predict next value by:
1. Fuzzify the last k values (where k = order)
2. Look up the matching FLRG
3. If no exact match, use nearest neighbor approach
4. Defuzzify by computing average center of consequent fuzzy sets

### 7. Defuzzification

We use the centroid defuzzification method:
- For FLRG with consequents {A_i1, A_i2, ..., A_im}
- Output = (center(A_i1) + center(A_i2) + ... + center(A_im)) / m
"""


def generate_complete_report(all_results: Dict, output_dir: str):
    """Generate a complete text-based report."""

    report_path = os.path.join(output_dir, "FULL_REPORT.txt")

    with open(report_path, 'w') as f:
        # Header
        f.write("="*80 + "\n")
        f.write(" " * 20 + "FUZZY TIME SERIES FORECASTING\n")
        f.write(" " * 25 + "PROJECT REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(
            f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Executive Summary
        f.write("="*80 + "\n")
        f.write("EXECUTIVE SUMMARY\n")
        f.write("="*80 + "\n\n")

        f.write(
            "This report presents the implementation and analysis of Fuzzy Time Series (FTS)\n")
        f.write("forecasting models applied to two distinct datasets:\n\n")
        f.write(
            "1. Mackey-Glass Time Series - A chaotic, non-linear benchmark dataset\n")
        f.write(
            "2. Influenza Specimens Data - Real-world epidemiological time series\n\n")

        f.write("Key Findings:\n")
        f.write("-" * 40 + "\n")

        for name, runner in all_results.items():
            if runner.best_result:
                br = runner.best_result
                f.write(f"\n{name}:\n")
                f.write(f"  Best Configuration: Order={br.config.order}, ")
                f.write(f"Partitions={br.config.num_partitions}, ")
                f.write(f"MF={br.config.mf_type.value}\n")
                f.write(f"  Test RMSE: {br.test_metrics['RMSE']:.6f}\n")

        # Methodology
        f.write("\n\n" + "="*80 + "\n")
        f.write("METHODOLOGY\n")
        f.write("="*80 + "\n")
        f.write(generate_methodology_section())

        # Results for each dataset
        for name, runner in all_results.items():
            f.write("\n\n" + "="*80 + "\n")
            f.write(f"RESULTS: {name.upper()}\n")
            f.write("="*80 + "\n\n")

            # Data characteristics
            f.write("Data Characteristics:\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Total samples: {len(runner.data)}\n")
            f.write(f"  Training samples: {runner.train_size}\n")
            f.write(f"  Testing samples: {len(runner.test_data)}\n")
            f.write(
                f"  Value range: [{runner.data.min():.4f}, {runner.data.max():.4f}]\n")
            f.write(f"  Mean: {runner.data.mean():.4f}\n")
            f.write(f"  Std Dev: {runner.data.std():.4f}\n\n")

            # Best configuration
            if runner.best_result:
                br = runner.best_result
                f.write("Best Configuration:\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Order: {br.config.order}")
                if br.config.order == 1:
                    f.write(" (First-Order FTS)\n")
                else:
                    f.write(f" (High-Order FTS, order {br.config.order})\n")
                f.write(
                    f"  Number of Partitions: {br.config.num_partitions}\n")
                f.write(
                    f"  Membership Function: {br.config.mf_type.value.capitalize()}\n")
                f.write(f"  Number of FLRGs: {len(br.model.flrgs)}\n\n")

                f.write("Performance Metrics:\n")
                f.write("-" * 40 + "\n")
                f.write(f"{'Metric':<15} {'Training':>15} {'Testing':>15}\n")
                f.write(f"{'-'*15} {'-'*15} {'-'*15}\n")
                f.write(
                    f"{'RMSE':<15} {br.train_metrics['RMSE']:>15.6f} {br.test_metrics['RMSE']:>15.6f}\n")
                f.write(
                    f"{'MAE':<15} {br.train_metrics['MAE']:>15.6f} {br.test_metrics['MAE']:>15.6f}\n")
                f.write(
                    f"{'MAPE (%)':<15} {br.train_metrics['MAPE']:>15.2f} {br.test_metrics['MAPE']:>15.2f}\n\n")

            # Order comparison
            f.write("First-Order vs High-Order Comparison:\n")
            f.write("-" * 40 + "\n")
            best_by_order = runner.get_best_by_order()
            f.write(f"{'Order':<10} {'RMSE':>12} {'MAE':>12} {'MAPE(%)':>12}\n")
            f.write(f"{'-'*10} {'-'*12} {'-'*12} {'-'*12}\n")
            for order in sorted(best_by_order.keys()):
                r = best_by_order[order]
                f.write(f"{order:<10} {r.test_metrics['RMSE']:>12.6f} ")
                f.write(
                    f"{r.test_metrics['MAE']:>12.6f} {r.test_metrics['MAPE']:>12.2f}\n")

            # MF type comparison
            f.write("\nMembership Function Type Comparison:\n")
            f.write("-" * 40 + "\n")
            best_by_mf = runner.get_best_by_mf_type()
            f.write(f"{'MF Type':<15} {'RMSE':>12} {'MAE':>12} {'MAPE(%)':>12}\n")
            f.write(f"{'-'*15} {'-'*12} {'-'*12} {'-'*12}\n")
            for mf_type in sorted(best_by_mf.keys()):
                r = best_by_mf[mf_type]
                f.write(f"{mf_type:<15} {r.test_metrics['RMSE']:>12.6f} ")
                f.write(
                    f"{r.test_metrics['MAE']:>12.6f} {r.test_metrics['MAPE']:>12.2f}\n")

        # Conclusions
        f.write("\n\n" + "="*80 + "\n")
        f.write("CONCLUSIONS\n")
        f.write("="*80 + "\n\n")

        f.write("1. Model Order Selection:\n")
        f.write("   - Higher-order models generally capture more complex patterns\n")
        f.write("   - However, too high an order may lead to overfitting\n")
        f.write("   - Optimal order varies by dataset characteristics\n\n")

        f.write("2. Partition Selection:\n")
        f.write("   - More partitions provide finer granularity but may overfit\n")
        f.write("   - Fewer partitions are more generalizable but less precise\n")
        f.write("   - Sweet spot typically between 7-15 partitions\n\n")

        f.write("3. Membership Function Selection:\n")
        f.write("   - Triangular MFs provide simple, interpretable results\n")
        f.write("   - Gaussian MFs often perform better for smooth data\n")
        f.write("   - Choice depends on data characteristics\n\n")

        f.write("="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")

    print(f"Full report saved to: {report_path}")
    return report_path


def save_best_model_config(result, output_path: str):
    """Save the best model configuration to a JSON file."""
    config = {
        'order': result.config.order,
        'num_partitions': result.config.num_partitions,
        'mf_type': result.config.mf_type.value,
        'margin_percent': result.config.margin_percent,
        'train_metrics': result.train_metrics,
        'test_metrics': result.test_metrics,
        'num_flrgs': len(result.model.flrgs),
        'universe': {
            'lower_bound': result.model.universe.lower_bound,
            'upper_bound': result.model.universe.upper_bound
        },
        'fuzzy_sets': result.model.get_fuzzy_sets_info()
    }

    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)


def generate_all_appendices(all_results: Dict, output_dir: str):
    """Generate all appendix files."""

    appendix_dir = os.path.join(output_dir, "appendices")
    os.makedirs(appendix_dir, exist_ok=True)

    for name, runner in all_results.items():
        if runner.best_result:
            dataset_name = name.lower().replace(' ', '_')

            # Generate FLRG appendix
            generate_flrg_appendix(
                runner.best_result.model,
                os.path.join(appendix_dir, f"{dataset_name}_flrgs.txt")
            )

            # Generate fuzzy sets appendix
            generate_fuzzy_sets_appendix(
                runner.best_result.model,
                os.path.join(appendix_dir, f"{dataset_name}_fuzzy_sets.txt")
            )

            # Save best configuration
            save_best_model_config(
                runner.best_result,
                os.path.join(appendix_dir, f"{dataset_name}_best_config.json")
            )

    print(f"Appendices saved to: {appendix_dir}")
