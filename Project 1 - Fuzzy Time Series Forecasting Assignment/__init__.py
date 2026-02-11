"""
Fuzzy Time Series Forecasting Package
=====================================

A complete implementation of First-Order and High-Order Fuzzy Time Series
forecasting methods from scratch.

Modules:
- fts_core: Core FTS algorithms and data structures
- fts_data: Data loading and preprocessing utilities
- fts_experiments: Experiment runner and parameter tuning
- fts_visualization: Plotting and visualization tools
- fts_report: Report generation utilities

    Author: Navid
    Course: Fuzzy Sets and Systems
"""

from fts_core import (
    FuzzyTimeSeries,
    FuzzySet,
    FuzzyLogicalRelation,
    FuzzyLogicalRelationGroup,
    UniverseOfDiscourse,
    FuzzySetPartitioner,
    MembershipFunctionType,
    FTSMetrics
)

from fts_data import (
    DataLoader,
    load_datasets,
    describe_dataset
)

from fts_experiments import (
    ExperimentRunner,
    ExperimentConfig,
    ExperimentResult,
    MultiDatasetExperiment
)

from fts_visualization import (
    FTSVisualizer,
    save_all_plots
)

__all__ = [
    'FuzzyTimeSeries',
    'FuzzySet',
    'FuzzyLogicalRelation',
    'FuzzyLogicalRelationGroup',
    'UniverseOfDiscourse',
    'FuzzySetPartitioner',
    'MembershipFunctionType',
    'FTSMetrics',
    'DataLoader',
    'load_datasets',
    'describe_dataset',
    'ExperimentRunner',
    'ExperimentConfig',
    'ExperimentResult',
    'MultiDatasetExperiment',
    'FTSVisualizer',
    'save_all_plots',
]
