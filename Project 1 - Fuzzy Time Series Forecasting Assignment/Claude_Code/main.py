"""
Fuzzy Time Series - Main Execution Script
==========================================
This script runs the complete FTS analysis on both datasets:
1. Mackey-Glass Time Series
2. Influenza Specimens Data (Total Specimens, Influenza A, Influenza B)

It performs systematic parameter tuning, generates reports, and saves results.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os
import sys
import json
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fts_core import FuzzyTimeSeries, MembershipFunctionType, FTSMetrics
from fts_data import DataLoader, load_datasets, describe_dataset
from fts_experiments import ExperimentRunner, MultiDatasetExperiment
from fts_visualization import FTSVisualizer


# Configuration
MACKEY_GLASS_PATH = "/mnt/user-data/uploads/mackey_glass.csv"
INFLUENZA_PATH = "/mnt/user-data/uploads/Specimens-Train.xlsx"
OUTPUT_DIR = "/home/claude/fuzzy_time_series/results"

# Parameter search space
ORDERS = [1, 2, 3, 4, 5]
PARTITIONS = [5, 7, 9, 11, 13, 15, 17]
MF_TYPES = [
    MembershipFunctionType.TRIANGULAR,
    MembershipFunctionType.GAUSSIAN,
    MembershipFunctionType.TRAPEZOIDAL,
    MembershipFunctionType.BELL
]
TRAIN_RATIO = 0.8


def run_analysis():
    """Run the complete FTS analysis on all datasets."""
    print("="*70)
    print("FUZZY TIME SERIES FORECASTING PROJECT")
    print("="*70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output Directory: {OUTPUT_DIR}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load all datasets
    print("\n" + "="*70)
    print("LOADING DATASETS")
    print("="*70)
    
    # Load Mackey-Glass
    print("\nLoading Mackey-Glass dataset...")
    mackey_glass_data = DataLoader.load_mackey_glass(MACKEY_GLASS_PATH)
    describe_dataset(mackey_glass_data, "Mackey-Glass")
    
    # Load Influenza data
    print("\nLoading Influenza dataset...")
    influenza_data = DataLoader.load_influenza_data(INFLUENZA_PATH)
    describe_dataset(influenza_data['total_specimens'], "Total Specimens")
    describe_dataset(influenza_data['influenza_a'], "Influenza A")
    describe_dataset(influenza_data['influenza_b'], "Influenza B")
    
    # All results storage
    all_results = {}
    
    # =========================================================================
    # DATASET 1: Mackey-Glass
    # =========================================================================
    print("\n" + "#"*70)
    print("# DATASET 1: MACKEY-GLASS TIME SERIES")
    print("#"*70)
    
    runner_mg = ExperimentRunner(
        data=mackey_glass_data,
        train_ratio=TRAIN_RATIO,
        dataset_name="Mackey-Glass"
    )
    
    runner_mg.run_grid_search(
        orders=ORDERS,
        partitions=PARTITIONS,
        mf_types=MF_TYPES,
        verbose=True
    )
    
    runner_mg.print_summary()
    runner_mg.save_results(os.path.join(OUTPUT_DIR, "mackey_glass"))
    all_results['Mackey-Glass'] = runner_mg
    
    # =========================================================================
    # DATASET 2: Total Specimens
    # =========================================================================
    print("\n" + "#"*70)
    print("# DATASET 2: TOTAL SPECIMENS")
    print("#"*70)
    
    runner_ts = ExperimentRunner(
        data=influenza_data['total_specimens'],
        train_ratio=TRAIN_RATIO,
        dataset_name="Total Specimens"
    )
    
    runner_ts.run_grid_search(
        orders=ORDERS,
        partitions=PARTITIONS,
        mf_types=MF_TYPES,
        verbose=True
    )
    
    runner_ts.print_summary()
    runner_ts.save_results(os.path.join(OUTPUT_DIR, "total_specimens"))
    all_results['Total Specimens'] = runner_ts
    
    # =========================================================================
    # DATASET 3: Influenza A
    # =========================================================================
    print("\n" + "#"*70)
    print("# DATASET 3: INFLUENZA A")
    print("#"*70)
    
    runner_a = ExperimentRunner(
        data=influenza_data['influenza_a'],
        train_ratio=TRAIN_RATIO,
        dataset_name="Influenza A"
    )
    
    runner_a.run_grid_search(
        orders=ORDERS,
        partitions=PARTITIONS,
        mf_types=MF_TYPES,
        verbose=True
    )
    
    runner_a.print_summary()
    runner_a.save_results(os.path.join(OUTPUT_DIR, "influenza_a"))
    all_results['Influenza A'] = runner_a
    
    # =========================================================================
    # DATASET 4: Influenza B
    # =========================================================================
    print("\n" + "#"*70)
    print("# DATASET 4: INFLUENZA B")
    print("#"*70)
    
    runner_b = ExperimentRunner(
        data=influenza_data['influenza_b'],
        train_ratio=TRAIN_RATIO,
        dataset_name="Influenza B"
    )
    
    runner_b.run_grid_search(
        orders=ORDERS,
        partitions=PARTITIONS,
        mf_types=MF_TYPES,
        verbose=True
    )
    
    runner_b.print_summary()
    runner_b.save_results(os.path.join(OUTPUT_DIR, "influenza_b"))
    all_results['Influenza B'] = runner_b
    
    # =========================================================================
    # GENERATE COMPREHENSIVE REPORT
    # =========================================================================
    generate_comprehensive_report(all_results, OUTPUT_DIR)
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("FINAL SUMMARY - ALL DATASETS")
    print("="*70)
    
    summary_data = []
    for name, runner in all_results.items():
        if runner.best_result:
            summary_data.append({
                'Dataset': name,
                'Best Order': runner.best_result.config.order,
                'Best Partitions': runner.best_result.config.num_partitions,
                'Best MF Type': runner.best_result.config.mf_type.value,
                'Test RMSE': runner.best_result.test_metrics['RMSE'],
                'Test MAE': runner.best_result.test_metrics['MAE'],
                'Test MAPE (%)': runner.best_result.test_metrics['MAPE']
            })
    
    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False))
    
    # Save final summary
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "final_summary.csv"), index=False)
    
    print(f"\n{'='*70}")
    print(f"Analysis complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    
    return all_results


def generate_comprehensive_report(all_results, output_dir):
    """Generate a comprehensive markdown report."""
    
    report_path = os.path.join(output_dir, "COMPREHENSIVE_REPORT.md")
    
    with open(report_path, 'w') as f:
        f.write("# Fuzzy Time Series Forecasting - Comprehensive Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Executive Summary\n\n")
        f.write("This report presents the results of applying Fuzzy Time Series (FTS) forecasting ")
        f.write("to multiple datasets using both First-Order FTS (FOFTS) and High-Order FTS (HOFTS) ")
        f.write("with various membership function types.\n\n")
        
        # Best configurations table
        f.write("### Best Configurations by Dataset\n\n")
        f.write("| Dataset | Order | Partitions | MF Type | Test RMSE | Test MAE | Test MAPE (%) |\n")
        f.write("|---------|-------|------------|---------|-----------|----------|---------------|\n")
        
        for name, runner in all_results.items():
            if runner.best_result:
                br = runner.best_result
                f.write(f"| {name} | {br.config.order} | {br.config.num_partitions} | ")
                f.write(f"{br.config.mf_type.value} | {br.test_metrics['RMSE']:.6f} | ")
                f.write(f"{br.test_metrics['MAE']:.6f} | {br.test_metrics['MAPE']:.2f} |\n")
        
        f.write("\n---\n\n")
        
        # Detailed analysis for each dataset
        for name, runner in all_results.items():
            f.write(f"## Dataset: {name}\n\n")
            
            # Data summary
            f.write("### Data Summary\n\n")
            f.write(f"- **Training samples:** {runner.train_size}\n")
            f.write(f"- **Testing samples:** {len(runner.test_data)}\n")
            f.write(f"- **Total samples:** {len(runner.data)}\n")
            f.write(f"- **Data range:** [{runner.data.min():.4f}, {runner.data.max():.4f}]\n\n")
            
            # Best configuration details
            if runner.best_result:
                br = runner.best_result
                f.write("### Best Model Configuration\n\n")
                f.write(f"- **Order:** {br.config.order} ")
                f.write(f"({'FOFTS' if br.config.order == 1 else f'HOFTS (order {br.config.order})'})\n")
                f.write(f"- **Number of Partitions:** {br.config.num_partitions}\n")
                f.write(f"- **Membership Function:** {br.config.mf_type.value.capitalize()}\n\n")
                
                f.write("### Performance Metrics (Best Model)\n\n")
                f.write("| Metric | Training | Testing |\n")
                f.write("|--------|----------|----------|\n")
                f.write(f"| RMSE | {br.train_metrics['RMSE']:.6f} | {br.test_metrics['RMSE']:.6f} |\n")
                f.write(f"| MAE | {br.train_metrics['MAE']:.6f} | {br.test_metrics['MAE']:.6f} |\n")
                f.write(f"| MAPE (%) | {br.train_metrics['MAPE']:.2f} | {br.test_metrics['MAPE']:.2f} |\n\n")
            
            # Order comparison
            f.write("### First-Order vs High-Order Comparison\n\n")
            comparison_df = runner.compare_orders()
            f.write(comparison_df.to_markdown(index=False))
            f.write("\n\n")
            
            # FLRGs for best model
            if runner.best_result:
                f.write("### Sample FLRGs (Best Model)\n\n")
                f.write("```\n")
                flrgs = runner.best_result.model.get_flrgs_as_strings()
                for flrg in flrgs[:20]:  # Show first 20
                    f.write(f"{flrg}\n")
                if len(flrgs) > 20:
                    f.write(f"... and {len(flrgs) - 20} more FLRGs\n")
                f.write("```\n\n")
            
            f.write("---\n\n")
        
        # Methodology section
        f.write("## Methodology\n\n")
        f.write("### Fuzzy Set Partitioning\n\n")
        f.write("The universe of discourse is partitioned into fuzzy sets using the following ")
        f.write("membership function types:\n\n")
        f.write("1. **Triangular:** Classic triangular membership functions with 50% overlap\n")
        f.write("2. **Trapezoidal:** Trapezoidal functions with flat top regions\n")
        f.write("3. **Gaussian:** Bell-shaped Gaussian curves\n")
        f.write("4. **Generalized Bell:** Adjustable bell-shaped functions\n\n")
        
        f.write("### FLRG Generation\n\n")
        f.write("For a model of order k, FLRGs are generated as:\n")
        f.write("- **First-Order (k=1):** F(t-1) → F(t)\n")
        f.write("- **High-Order (k≥2):** (F(t-k), F(t-k+1), ..., F(t-1)) → F(t)\n\n")
        
        f.write("### Defuzzification\n\n")
        f.write("The centroid method is used for defuzzification, computing the average ")
        f.write("center of all consequent fuzzy sets in the matched FLRG.\n\n")
        
        f.write("### Performance Metrics\n\n")
        f.write("- **RMSE:** Root Mean Square Error = √(mean((actual - predicted)²))\n")
        f.write("- **MAE:** Mean Absolute Error = mean(|actual - predicted|)\n")
        f.write("- **MAPE:** Mean Absolute Percentage Error = mean(|actual - predicted| / |actual|) × 100\n\n")
        
        f.write("---\n\n")
        f.write("## Visualizations\n\n")
        f.write("The following visualizations are available in the results directory:\n\n")
        f.write("- Membership function plots for each dataset\n")
        f.write("- Actual vs Predicted time series plots\n")
        f.write("- Error metric heatmaps across parameter space\n")
        f.write("- Fuzzification visualization\n\n")
    
    print(f"\nComprehensive report saved to: {report_path}")


def run_demo_prediction():
    """Run a demonstration of the prediction interface."""
    from fts_interface import FTSPredictor, create_prediction_interface
    
    print("\n" + "="*70)
    print("PREDICTION INTERFACE DEMONSTRATION")
    print("="*70)
    
    # Load Mackey-Glass data for demo
    data = DataLoader.load_mackey_glass(MACKEY_GLASS_PATH)
    train_data = data[:int(len(data) * 0.8)]
    
    # Create predictor with best-found configuration
    predictor = FTSPredictor()
    predictor.load_model_from_config(
        training_data=train_data,
        order=3,
        num_partitions=11,
        mf_type="triangular"
    )
    
    # Show model info
    info = predictor.get_model_info()
    print(f"\nModel loaded with:")
    print(f"  Order: {info['order']}")
    print(f"  Partitions: {info['num_partitions']}")
    print(f"  MF Type: {info['mf_type']}")
    print(f"  Number of FLRGs: {info['num_flrgs']}")
    
    # Demo predictions
    print("\n--- Demo Predictions ---")
    
    # Use last values from training data
    test_values = list(train_data[-10:])
    print(f"\nInput values: {[f'{v:.4f}' for v in test_values]}")
    
    result = predictor.predict_next(test_values)
    print(f"Fuzzified as: {result['fuzzified_input']}")
    print(f"Matched FLRG: {result['matched_flrg']}")
    print(f"Predicted next value: {result['prediction']:.6f}")
    
    # Multi-step forecast
    print("\n--- Multi-step Forecast (5 steps) ---")
    results = predictor.predict_multiple(test_values, steps=5)
    for r in results:
        print(f"  Step {r['step']}: {r['prediction']:.6f}")


if __name__ == "__main__":
    # Run the main analysis
    results = run_analysis()
    
    # Run prediction demo
    run_demo_prediction()
    
    print("\n" + "="*70)
    print("ALL TASKS COMPLETED SUCCESSFULLY")
    print("="*70)
