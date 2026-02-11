import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from itertools import product
import json
import os
from datetime import datetime

from fts_core import (
    FuzzyTimeSeries, MembershipFunctionType, FTSMetrics
)
from fts_visualization import FTSVisualizer


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    order: int
    num_partitions: int
    mf_type: MembershipFunctionType
    margin_percent: float = 0.1

    def to_dict(self) -> Dict:
        return {
            'order': self.order,
            'partitions': self.num_partitions,
            'mf_type': self.mf_type.value,
            'margin_percent': self.margin_percent
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'ExperimentConfig':
        return cls(
            order=d['order'],
            num_partitions=d['partitions'],
            mf_type=MembershipFunctionType(d['mf_type']),
            margin_percent=d.get('margin_percent', 0.1)
        )


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    config: ExperimentConfig
    train_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    train_predictions: np.ndarray
    test_predictions: np.ndarray
    model: FuzzyTimeSeries
    execution_time: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization (excluding large arrays and model)."""
        return {
            **self.config.to_dict(),
            'train_RMSE': self.train_metrics['RMSE'],
            'train_MAE': self.train_metrics['MAE'],
            'train_MAPE': self.train_metrics['MAPE'],
            'test_RMSE': self.test_metrics['RMSE'],
            'test_MAE': self.test_metrics['MAE'],
            'test_MAPE': self.test_metrics['MAPE'],
            'execution_time': self.execution_time
        }

    def get_summary_dict(self) -> Dict:
        """Get a simplified summary for display."""
        return {
            'order': self.config.order,
            'partitions': self.config.num_partitions,
            'mf_type': self.config.mf_type.value,
            'RMSE': self.test_metrics['RMSE'],
            'MAE': self.test_metrics['MAE'],
            'MAPE': self.test_metrics['MAPE']
        }


class ExperimentRunner:
    """
    Runs systematic experiments with different FTS configurations.
    """

    def __init__(
        self,
        data: np.ndarray,
        train_ratio: float = 0.8,
        dataset_name: str = "Dataset"
    ):
        """
        Initialize the experiment runner.

        Args:
            data: Time series data
            train_ratio: Ratio of data to use for training (default 80%)
            dataset_name: Name of the dataset for labeling
        """
        self.data = np.array(data).flatten()
        self.train_ratio = train_ratio
        self.dataset_name = dataset_name

        # Split data
        self.train_size = int(len(self.data) * train_ratio)
        self.train_data = self.data[:self.train_size]
        self.test_data = self.data[self.train_size:]

        # Storage for results
        self.results: List[ExperimentResult] = []
        self.best_result: Optional[ExperimentResult] = None

    def run_single_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        """
        Run a single experiment with the given configuration.

        Args:
            config: Experiment configuration

        Returns:
            ExperimentResult with all metrics and predictions
        """
        import time
        start_time = time.time()

        # Create and train model
        model = FuzzyTimeSeries(
            order=config.order,
            num_partitions=config.num_partitions,
            mf_type=config.mf_type,
            margin_percent=config.margin_percent
        )
        model.fit(self.train_data)

        # Generate predictions for training data
        train_predictions = model.predict(self.train_data)

        # Generate predictions for test data
        # For test data, we use training data as initial history
        test_predictions = np.full(len(self.test_data), np.nan)

        for i in range(len(self.test_data)):
            if i < config.order:
                # Use training data for initial predictions
                history = list(
                    self.train_data[-(config.order - i):]) + list(self.test_data[:i])
            else:
                history = list(self.test_data[:i])

            if len(history) >= config.order:
                test_predictions[i] = model.predict_next(history)

        execution_time = time.time() - start_time

        # Calculate metrics
        train_metrics = FTSMetrics.all_metrics(
            self.train_data[config.order:],  # Skip first 'order' values
            train_predictions[config.order:]
        )

        test_metrics = FTSMetrics.all_metrics(
            self.test_data,
            test_predictions
        )

        return ExperimentResult(
            config=config,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            train_predictions=train_predictions,
            test_predictions=test_predictions,
            model=model,
            execution_time=execution_time
        )

    def run_grid_search(
        self,
        orders: List[int] = [1, 2, 3, 4, 5],
        partitions: List[int] = [5, 7, 9, 11, 13, 15],
        mf_types: List[MembershipFunctionType] = None,
        verbose: bool = True
    ) -> List[ExperimentResult]:
        """
        Run a grid search over parameter space.

        Args:
            orders: List of orders to test
            partitions: List of partition counts to test
            mf_types: List of membership function types to test
            verbose: Whether to print progress

        Returns:
            List of all experiment results
        """
        if mf_types is None:
            mf_types = [
                MembershipFunctionType.TRIANGULAR,
                MembershipFunctionType.GAUSSIAN,
                MembershipFunctionType.TRAPEZOIDAL,
                MembershipFunctionType.BELL
            ]

        total_experiments = len(orders) * len(partitions) * len(mf_types)

        if verbose:
            print(f"\n{'='*60}")
            print(f"Grid Search for {self.dataset_name}")
            print(f"{'='*60}")
            print(f"Total experiments: {total_experiments}")
            print(f"Orders: {orders}")
            print(f"Partitions: {partitions}")
            print(f"MF Types: {[mf.value for mf in mf_types]}")
            print(f"{'='*60}\n")

        self.results = []

        for idx, (order, num_partitions, mf_type) in enumerate(product(orders, partitions, mf_types)):
            config = ExperimentConfig(
                order=order,
                num_partitions=num_partitions,
                mf_type=mf_type
            )

            try:
                result = self.run_single_experiment(config)
                self.results.append(result)

                if verbose:
                    print(f"[{idx+1}/{total_experiments}] Order={order}, "
                          f"Partitions={num_partitions}, MF={mf_type.value}: "
                          f"Test RMSE={result.test_metrics['RMSE']:.6f}")
            except Exception as e:
                if verbose:
                    print(f"[{idx+1}/{total_experiments}] ERROR: {e}")

        # Find best result
        self._find_best_result()

        return self.results

    def _find_best_result(self, metric: str = 'RMSE'):
        """Find the best result based on test metrics."""
        if not self.results:
            return

        # Filter out results with NaN metrics
        valid_results = [r for r in self.results
                         if not np.isnan(r.test_metrics.get(metric, np.nan))]

        if valid_results:
            self.best_result = min(valid_results,
                                   key=lambda r: r.test_metrics[metric])

    def get_results_dataframe(self) -> pd.DataFrame:
        """Convert results to a pandas DataFrame for analysis."""
        if not self.results:
            return pd.DataFrame()

        data = [r.to_dict() for r in self.results]
        df = pd.DataFrame(data)
        return df.sort_values('test_RMSE').reset_index(drop=True)

    def get_best_by_mf_type(self) -> Dict[str, ExperimentResult]:
        """Get the best result for each membership function type."""
        best_by_type = {}

        for result in self.results:
            mf_type = result.config.mf_type.value
            if mf_type not in best_by_type or \
               result.test_metrics['RMSE'] < best_by_type[mf_type].test_metrics['RMSE']:
                best_by_type[mf_type] = result

        return best_by_type

    def get_best_by_order(self) -> Dict[int, ExperimentResult]:
        """Get the best result for each order."""
        best_by_order = {}

        for result in self.results:
            order = result.config.order
            if order not in best_by_order or \
               result.test_metrics['RMSE'] < best_by_order[order].test_metrics['RMSE']:
                best_by_order[order] = result

        return best_by_order

    def compare_orders(self) -> pd.DataFrame:
        """Generate a comparison table of FOFTS vs HOFTS performance."""
        best_by_order = self.get_best_by_order()

        data = []
        for order, result in sorted(best_by_order.items()):
            data.append({
                'Order': order,
                'Type': 'FOFTS' if order == 1 else f'HOFTS (order={order})',
                'Partitions': result.config.num_partitions,
                'MF Type': result.config.mf_type.value,
                'Test RMSE': result.test_metrics['RMSE'],
                'Test MAE': result.test_metrics['MAE'],
                'Test MAPE': result.test_metrics['MAPE']
            })

        return pd.DataFrame(data)

    def print_summary(self):
        """Print a summary of the experiment results."""
        if not self.results:
            print("No results available.")
            return

        print(f"\n{'='*70}")
        print(f"EXPERIMENT SUMMARY: {self.dataset_name}")
        print(f"{'='*70}")
        print(f"Total experiments run: {len(self.results)}")
        print(f"Training samples: {self.train_size}")
        print(f"Testing samples: {len(self.test_data)}")

        if self.best_result:
            print(f"\n{'='*70}")
            print("BEST CONFIGURATION:")
            print(f"{'='*70}")
            print(f"  Order: {self.best_result.config.order}")
            print(f"  Partitions: {self.best_result.config.num_partitions}")
            print(f"  MF Type: {self.best_result.config.mf_type.value}")
            print(f"\n  Test Metrics:")
            print(f"    RMSE: {self.best_result.test_metrics['RMSE']:.6f}")
            print(f"    MAE: {self.best_result.test_metrics['MAE']:.6f}")
            print(f"    MAPE: {self.best_result.test_metrics['MAPE']:.2f}%")

        # Compare orders
        print(f"\n{'='*70}")
        print("COMPARISON BY ORDER (First-Order vs High-Order):")
        print(f"{'='*70}")
        comparison_df = self.compare_orders()
        print(comparison_df.to_string(index=False))

        # Best by MF type
        print(f"\n{'='*70}")
        print("BEST BY MEMBERSHIP FUNCTION TYPE:")
        print(f"{'='*70}")
        best_by_mf = self.get_best_by_mf_type()
        for mf_type, result in sorted(best_by_mf.items()):
            print(f"  {mf_type:12s}: RMSE={result.test_metrics['RMSE']:.6f} "
                  f"(order={result.config.order}, partitions={result.config.num_partitions})")

    def save_results(self, output_dir: str, prefix: str = ""):
        """Save experiment results and visualizations."""
        os.makedirs(output_dir, exist_ok=True)

        # Save results DataFrame
        df = self.get_results_dataframe()
        df.to_csv(os.path.join(
            output_dir, f"{prefix}results.csv"), index=False)

        # Save summary
        summary = {
            'dataset_name': self.dataset_name,
            'train_size': self.train_size,
            'test_size': len(self.test_data),
            'train_ratio': self.train_ratio,
            'num_experiments': len(self.results),
            'best_config': self.best_result.config.to_dict() if self.best_result else None,
            'best_metrics': self.best_result.test_metrics if self.best_result else None,
            'timestamp': datetime.now().isoformat()
        }

        with open(os.path.join(output_dir, f"{prefix}summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)

        # Generate visualizations
        visualizer = FTSVisualizer()

        if self.best_result:
            # Plot membership functions for best model
            visualizer.plot_membership_functions(
                self.best_result.model.fuzzy_sets,
                self.best_result.model.universe,
                title=f"Best Model Membership Functions - {self.dataset_name}",
                save_path=os.path.join(
                    output_dir, f"{prefix}best_membership_functions.png")
            )

            # Plot predictions
            all_actual = np.concatenate([self.train_data, self.test_data])
            all_predicted = np.concatenate([
                self.best_result.train_predictions,
                self.best_result.test_predictions
            ])

            visualizer.plot_time_series_comparison(
                all_actual, all_predicted,
                title=f"Best Model Predictions - {self.dataset_name}",
                train_end_idx=self.train_size,
                save_path=os.path.join(
                    output_dir, f"{prefix}best_predictions.png")
            )

        # Plot parameter heatmaps for each MF type
        results_dicts = [r.get_summary_dict() for r in self.results]

        for mf_type in set(r['mf_type'] for r in results_dicts):
            mf_results = [r for r in results_dicts if r['mf_type'] == mf_type]

            visualizer.plot_parameter_heatmap(
                mf_results,
                x_param='partitions',
                y_param='order',
                metric='RMSE',
                title=f"RMSE Heatmap ({mf_type}) - {self.dataset_name}",
                save_path=os.path.join(
                    output_dir, f"{prefix}heatmap_{mf_type}.png")
            )

        # Plot metrics comparison
        visualizer.plot_error_metrics_comparison(
            results_dicts,
            x_param='partitions',
            group_param='order',
            title=f"Performance Metrics - {self.dataset_name}",
            save_path=os.path.join(
                output_dir, f"{prefix}metrics_comparison.png")
        )

        import matplotlib.pyplot as plt
        plt.close('all')

        print(f"\nResults saved to: {output_dir}")


class MultiDatasetExperiment:
    """
    Runs experiments across multiple datasets and aggregates results.
    """

    def __init__(self):
        self.datasets: Dict[str, np.ndarray] = {}
        self.runners: Dict[str, ExperimentRunner] = {}

    def add_dataset(self, name: str, data: np.ndarray, train_ratio: float = 0.8):
        """Add a dataset for experimentation."""
        self.datasets[name] = np.array(data).flatten()
        self.runners[name] = ExperimentRunner(
            self.datasets[name],
            train_ratio=train_ratio,
            dataset_name=name
        )

    def run_all(
        self,
        orders: List[int] = [1, 2, 3, 4, 5],
        partitions: List[int] = [5, 7, 9, 11, 13, 15],
        mf_types: List[MembershipFunctionType] = None,
        output_dir: str = "results",
        verbose: bool = True
    ):
        """Run experiments on all datasets."""
        for name, runner in self.runners.items():
            print(f"\n{'#'*70}")
            print(f"# Running experiments for: {name}")
            print(f"{'#'*70}")

            runner.run_grid_search(
                orders=orders,
                partitions=partitions,
                mf_types=mf_types,
                verbose=verbose
            )

            runner.print_summary()

            # Save results
            dataset_output_dir = os.path.join(
                output_dir, name.replace(' ', '_'))
            runner.save_results(dataset_output_dir)

    def get_comparison_table(self) -> pd.DataFrame:
        """Generate a comparison table across all datasets."""
        data = []

        for name, runner in self.runners.items():
            if runner.best_result:
                data.append({
                    'Dataset': name,
                    'Best Order': runner.best_result.config.order,
                    'Best Partitions': runner.best_result.config.num_partitions,
                    'Best MF Type': runner.best_result.config.mf_type.value,
                    'Test RMSE': runner.best_result.test_metrics['RMSE'],
                    'Test MAE': runner.best_result.test_metrics['MAE'],
                    'Test MAPE': runner.best_result.test_metrics['MAPE']
                })

        return pd.DataFrame(data)
