import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import List, Dict, Optional, Tuple, Union
import os

# Import from core module
from fts_core import (
    FuzzyTimeSeries, FuzzySet, MembershipFunctionType,
    UniverseOfDiscourse, FTSMetrics
)


class FTSVisualizer:
    def __init__(self, figsize: Tuple[int, int] = (12, 8), style: str = 'seaborn-v0_8-whitegrid'):
        self.figsize = figsize
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')

        # Color palette for consistency
        self.colors = {
            'actual': '#2E86AB',
            'predicted': '#E94F37',
            'forecast': '#F39237',
            'train': '#44AF69',
            'test': '#A23B72',
            'error': '#C73E1D'
        }

    def plot_membership_functions(
        self,
        fuzzy_sets: List[FuzzySet],
        universe: UniverseOfDiscourse,
        title: str = "Fuzzy Set Membership Functions",
        save_path: Optional[str] = None,
        num_points: int = 500
    ) -> Figure:
        fig, ax = plt.subplots(figsize=self.figsize)

        # Generate x values across the universe
        x_range = np.linspace(
            universe.lower_bound() - universe.get_range() * 0.1,
            universe.upper_bound() + universe.get_range() * 0.1,
            num_points
        )

        # Color map for different fuzzy sets
        colors = plt.cm.tab20(np.linspace(0, 1, len(fuzzy_sets)))

        # Plot each membership function
        for idx, fs in enumerate(fuzzy_sets):
            y_values = [fs.membership(x) for x in x_range]
            ax.plot(x_range, y_values, label=fs.name,
                    color=colors[idx], linewidth=2)

            # Mark the center
            ax.axvline(
                x=fs.center, color=colors[idx], linestyle='--', alpha=0.3)

        # Add universe bounds
        ax.axvline(x=universe.lower_bound(), color='gray',
                   linestyle=':', alpha=0.5, label='Universe bounds')
        ax.axvline(x=universe.upper_bound(), color='gray',
                   linestyle=':', alpha=0.5)

        ax.set_xlabel('Value', fontsize=12)
        ax.set_ylabel('Membership Degree', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylim(-0.05, 1.1)
        ax.legend(loc='upper right', ncol=min(4, len(fuzzy_sets)))
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def plot_time_series_comparison(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        title: str = "Actual vs Predicted Time Series",
        train_end_idx: Optional[int] = None,
        save_path: Optional[str] = None
    ) -> Figure:
        fig, axes = plt.subplots(2, 1, figsize=(
            self.figsize[0], self.figsize[1] * 1.2))

        # Top plot: Time series comparison
        ax1 = axes[0]
        time_idx = np.arange(len(actual))

        ax1.plot(time_idx, actual, color=self.colors['actual'],
                 label='Actual', linewidth=1.5, alpha=0.8)
        ax1.plot(time_idx, predicted, color=self.colors['predicted'],
                 label='Predicted', linewidth=1.5, alpha=0.8)

        if train_end_idx is not None:
            ax1.axvline(x=train_end_idx, color='green', linestyle='--',
                        alpha=0.7, label='Train/Test Split')
            ax1.axvspan(0, train_end_idx, alpha=0.1,
                        color='green', label='Training')
            ax1.axvspan(train_end_idx, len(actual), alpha=0.1,
                        color='red', label='Testing')

        ax1.set_xlabel('Time Index', fontsize=12)
        ax1.set_ylabel('Value', fontsize=12)
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        # Bottom plot: Prediction error
        ax2 = axes[1]
        error = actual - predicted

        ax2.fill_between(time_idx, 0, error, where=error >= 0,
                         color='green', alpha=0.3, label='Positive Error')
        ax2.fill_between(time_idx, 0, error, where=error < 0,
                         color='red', alpha=0.3, label='Negative Error')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        if train_end_idx is not None:
            ax2.axvline(x=train_end_idx, color='green',
                        linestyle='--', alpha=0.7)

        ax2.set_xlabel('Time Index', fontsize=12)
        ax2.set_ylabel('Error (Actual - Predicted)', fontsize=12)
        ax2.set_title('Prediction Error', fontsize=12)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def plot_error_metrics_comparison(
        self,
        results: List[Dict],
        x_param: str = 'partitions',
        group_param: str = 'order',
        title: str = "Performance Metrics Across Parameter Space",
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Plot error metrics across different parameter configurations.

        Args:
            results: List of dictionaries with parameter values and metrics
            x_param: Parameter to use for x-axis
            group_param: Parameter to group by (different lines)
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib Figure object
        """
        fig, axes = plt.subplots(1, 3, figsize=(
            self.figsize[0] * 1.2, self.figsize[1] * 0.6))

        metrics = ['RMSE', 'MAE', 'MAPE']

        # Group results by group_param
        groups = {}
        for r in results:
            group_val = r[group_param]
            if group_val not in groups:
                groups[group_val] = []
            groups[group_val].append(r)

        # Sort each group by x_param
        for group_val in groups:
            groups[group_val] = sorted(
                groups[group_val], key=lambda x: x[x_param])

        # Plot each metric
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(groups)))

        for ax, metric in zip(axes, metrics):
            for idx, (group_val, group_results) in enumerate(sorted(groups.items())):
                x_vals = [r[x_param] for r in group_results]
                y_vals = [r[metric] for r in group_results]

                ax.plot(x_vals, y_vals, 'o-', label=f'{group_param}={group_val}',
                        color=colors[idx], linewidth=2, markersize=6)

            ax.set_xlabel(x_param.capitalize(), fontsize=11)
            ax.set_ylabel(metric, fontsize=11)
            ax.set_title(metric, fontsize=12, fontweight='bold')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def plot_forecast(
        self,
        historical: np.ndarray,
        forecast: np.ndarray,
        title: str = "Time Series Forecast",
        save_path: Optional[str] = None
    ) -> Figure:

        fig, ax = plt.subplots(figsize=self.figsize)

        hist_idx = np.arange(len(historical))
        forecast_idx = np.arange(
            len(historical), len(historical) + len(forecast))

        # Plot historical data
        ax.plot(hist_idx, historical, color=self.colors['actual'],
                label='Historical Data', linewidth=1.5)

        # Plot forecast
        ax.plot(forecast_idx, forecast, color=self.colors['forecast'],
                label='Forecast', linewidth=2, linestyle='--')

        # Add connection point
        ax.plot([len(historical) - 1, len(historical)],
                [historical[-1], forecast[0]],
                color=self.colors['forecast'], linestyle='--', linewidth=2)

        # Mark forecast start
        ax.axvline(x=len(historical) - 0.5, color='gray',
                   linestyle=':', alpha=0.7)
        ax.axvspan(len(historical) - 0.5, len(historical) + len(forecast),
                   alpha=0.1, color='yellow')

        ax.set_xlabel('Time Index', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def plot_fuzzification(
        self,
        data: np.ndarray,
        fuzzified: List[str],
        fuzzy_sets: List[FuzzySet],
        title: str = "Time Series Fuzzification",
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Plot the original time series with fuzzification overlay.

        Args:
            data: Original time series data
            fuzzified: List of fuzzy set names for each data point
            fuzzy_sets: List of fuzzy sets used
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 1, figsize=(
            self.figsize[0], self.figsize[1] * 1.2))

        # Create color map for fuzzy sets
        fs_names = [fs.name for fs in fuzzy_sets]
        colors = plt.cm.tab20(np.linspace(0, 1, len(fs_names)))
        color_map = {name: colors[i] for i, name in enumerate(fs_names)}

        # Top plot: Original time series with colored points
        ax1 = axes[0]
        ax1.plot(data, color='gray', alpha=0.5, linewidth=1)

        for i, (val, fuzz) in enumerate(zip(data, fuzzified)):
            ax1.scatter(i, val, color=color_map[fuzz], s=10, alpha=0.7)

        ax1.set_xlabel('Time Index', fontsize=12)
        ax1.set_ylabel('Value', fontsize=12)
        ax1.set_title(f'{title} - Colored by Fuzzy Set',
                      fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Create legend
        handles = [plt.Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=color_map[name], markersize=8, label=name)
                   for name in fs_names]
        ax1.legend(handles=handles, loc='upper right',
                   ncol=min(5, len(fs_names)))

        # Bottom plot: Fuzzy set assignments over time
        ax2 = axes[1]
        fs_indices = [fs_names.index(f) for f in fuzzified]

        ax2.scatter(range(len(fuzzified)), fs_indices,
                    c='blue', s=5, alpha=0.5)
        ax2.set_xlabel('Time Index', fontsize=12)
        ax2.set_ylabel('Fuzzy Set Index', fontsize=12)
        ax2.set_title('Fuzzy Set Assignment Over Time', fontsize=12)
        ax2.set_yticks(range(len(fs_names)))
        ax2.set_yticklabels(fs_names)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def plot_parameter_heatmap(
        self,
        results: List[Dict],
        x_param: str = 'partitions',
        y_param: str = 'order',
        metric: str = 'RMSE',
        title: str = None,
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Create a heatmap of performance metrics across two parameters.

        Args:
            results: List of dictionaries with parameter values and metrics
            x_param: Parameter for x-axis
            y_param: Parameter for y-axis
            metric: Metric to display in heatmap
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib Figure object
        """
        # Extract unique parameter values
        x_vals = sorted(set(r[x_param] for r in results))
        y_vals = sorted(set(r[y_param] for r in results))

        # Create matrix
        matrix = np.full((len(y_vals), len(x_vals)), np.nan)

        for r in results:
            x_idx = x_vals.index(r[x_param])
            y_idx = y_vals.index(r[y_param])
            matrix[y_idx, x_idx] = r[metric]

        fig, ax = plt.subplots(
            figsize=(self.figsize[0] * 0.8, self.figsize[1] * 0.8))

        # Create heatmap
        im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(metric, fontsize=12)

        # Set ticks and labels
        ax.set_xticks(range(len(x_vals)))
        ax.set_xticklabels([str(v) for v in x_vals])
        ax.set_yticks(range(len(y_vals)))
        ax.set_yticklabels([str(v) for v in y_vals])

        ax.set_xlabel(x_param.capitalize(), fontsize=12)
        ax.set_ylabel(y_param.capitalize(), fontsize=12)

        if title is None:
            title = f'{metric} Heatmap: {y_param} vs {x_param}'
        ax.set_title(title, fontsize=14, fontweight='bold')

        # Add value annotations
        for i in range(len(y_vals)):
            for j in range(len(x_vals)):
                if not np.isnan(matrix[i, j]):
                    text = ax.text(j, i, f'{matrix[i, j]:.3f}',
                                   ha='center', va='center', fontsize=9,
                                   color='white' if matrix[i, j] > np.nanmedian(matrix) else 'black')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig


def save_all_plots(visualizer: FTSVisualizer, model: FuzzyTimeSeries,
                   actual: np.ndarray, predicted: np.ndarray,
                   output_dir: str, prefix: str = ""):

    os.makedirs(output_dir, exist_ok=True)

    # Plot membership functions
    visualizer.plot_membership_functions(
        model.fuzzy_sets,
        model.universe,
        title=f"Membership Functions ({model.mf_type.value.capitalize()}, {model.num_partitions} partitions)",
        save_path=os.path.join(output_dir, f"{prefix}membership_functions.png")
    )

    # Plot time series comparison
    visualizer.plot_time_series_comparison(
        actual, predicted,
        title=f"Actual vs Predicted (Order={model.order})",
        save_path=os.path.join(
            output_dir, f"{prefix}time_series_comparison.png")
    )

    # Plot fuzzification
    visualizer.plot_fuzzification(
        model.training_data,
        model.fuzzified_series,
        model.fuzzy_sets,
        title="Training Data Fuzzification",
        save_path=os.path.join(output_dir, f"{prefix}fuzzification.png")
    )

    plt.close('all')
