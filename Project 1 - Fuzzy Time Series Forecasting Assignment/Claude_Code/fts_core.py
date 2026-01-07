"""
Fuzzy Time Series Core Module
=============================
This module implements the core algorithms for Fuzzy Time Series (FTS) forecasting,
including both First-Order FTS (FOFTS) and High-Order FTS (HOFTS).

All implementations are from scratch as per project requirements.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict


class MembershipFunctionType(Enum):
    """Enumeration of supported membership function types."""
    TRIANGULAR = "triangular"
    TRAPEZOIDAL = "trapezoidal"
    GAUSSIAN = "gaussian"
    BELL = "bell"


@dataclass
class FuzzySet:
    """
    Represents a single fuzzy set with its membership function.
    
    Attributes:
        name: Linguistic label for the fuzzy set (e.g., 'A1', 'Low', 'Medium')
        mf_type: Type of membership function
        parameters: Parameters defining the membership function shape
        center: The center/midpoint value of the fuzzy set for defuzzification
    """
    name: str
    mf_type: MembershipFunctionType
    parameters: Dict[str, float]
    center: float
    
    def membership(self, x: float) -> float:
        """
        Calculate the membership degree of a crisp value x in this fuzzy set.
        
        Args:
            x: The crisp input value
            
        Returns:
            Membership degree in range [0, 1]
        """
        if self.mf_type == MembershipFunctionType.TRIANGULAR:
            return self._triangular_membership(x)
        elif self.mf_type == MembershipFunctionType.TRAPEZOIDAL:
            return self._trapezoidal_membership(x)
        elif self.mf_type == MembershipFunctionType.GAUSSIAN:
            return self._gaussian_membership(x)
        elif self.mf_type == MembershipFunctionType.BELL:
            return self._bell_membership(x)
        else:
            raise ValueError(f"Unknown membership function type: {self.mf_type}")
    
    def _triangular_membership(self, x: float) -> float:
        """
        Triangular membership function.
        Parameters: a (left), b (peak), c (right)
        
              1 |     /\
                |    /  \
                |   /    \
              0 |__/______\__
                  a  b    c
        """
        a = self.parameters['a']  # Left point
        b = self.parameters['b']  # Peak point
        c = self.parameters['c']  # Right point
        
        if x <= a or x >= c:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a) if b != a else 1.0
        else:  # b < x < c
            return (c - x) / (c - b) if c != b else 1.0
    
    def _trapezoidal_membership(self, x: float) -> float:
        """
        Trapezoidal membership function.
        Parameters: a (left), b (left shoulder), c (right shoulder), d (right)
        
              1 |    ____
                |   /    \
                |  /      \
              0 |_/________\__
                 a b      c d
        """
        a = self.parameters['a']  # Left point
        b = self.parameters['b']  # Left shoulder
        c = self.parameters['c']  # Right shoulder
        d = self.parameters['d']  # Right point
        
        if x <= a or x >= d:
            return 0.0
        elif a < x < b:
            return (x - a) / (b - a) if b != a else 1.0
        elif b <= x <= c:
            return 1.0
        else:  # c < x < d
            return (d - x) / (d - c) if d != c else 1.0
    
    def _gaussian_membership(self, x: float) -> float:
        """
        Gaussian membership function.
        Parameters: c (center), sigma (standard deviation)
        
        μ(x) = exp(-((x - c)^2) / (2 * sigma^2))
        """
        c = self.parameters['c']
        sigma = self.parameters['sigma']
        
        return np.exp(-((x - c) ** 2) / (2 * sigma ** 2))
    
    def _bell_membership(self, x: float) -> float:
        """
        Generalized bell-shaped membership function.
        Parameters: a (width), b (slope), c (center)
        
        μ(x) = 1 / (1 + |((x - c) / a)|^(2b))
        """
        a = self.parameters['a']  # Width
        b = self.parameters['b']  # Slope
        c = self.parameters['c']  # Center
        
        if a == 0:
            return 1.0 if x == c else 0.0
        
        return 1 / (1 + np.abs((x - c) / a) ** (2 * b))


@dataclass
class UniverseOfDiscourse:
    """
    Represents the universe of discourse for a fuzzy variable.
    
    Attributes:
        min_val: Minimum value of the universe
        max_val: Maximum value of the universe
        margin: Additional margin added to accommodate future values
    """
    min_val: float
    max_val: float
    margin: float = 0.0
    
    @property
    def lower_bound(self) -> float:
        """Get the effective lower bound including margin."""
        return self.min_val - self.margin
    
    @property
    def upper_bound(self) -> float:
        """Get the effective upper bound including margin."""
        return self.max_val + self.margin
    
    @property
    def range(self) -> float:
        """Get the total range of the universe."""
        return self.upper_bound - self.lower_bound


class FuzzySetPartitioner:
    """
    Creates fuzzy sets by partitioning the universe of discourse.
    
    This class implements various strategies for creating fuzzy set partitions
    with different membership function types.
    """
    
    def __init__(
        self,
        universe: UniverseOfDiscourse,
        num_partitions: int,
        mf_type: MembershipFunctionType = MembershipFunctionType.TRIANGULAR,
        prefix: str = "A"
    ):
        """
        Initialize the partitioner.
        
        Args:
            universe: The universe of discourse to partition
            num_partitions: Number of fuzzy sets to create
            mf_type: Type of membership function to use
            prefix: Prefix for fuzzy set names (e.g., 'A' creates A1, A2, ...)
        """
        self.universe = universe
        self.num_partitions = num_partitions
        self.mf_type = mf_type
        self.prefix = prefix
        self.fuzzy_sets: List[FuzzySet] = []
        
    def create_partitions(self) -> List[FuzzySet]:
        """
        Create fuzzy set partitions based on the specified membership function type.
        
        Returns:
            List of FuzzySet objects covering the universe of discourse
        """
        if self.mf_type == MembershipFunctionType.TRIANGULAR:
            self.fuzzy_sets = self._create_triangular_partitions()
        elif self.mf_type == MembershipFunctionType.TRAPEZOIDAL:
            self.fuzzy_sets = self._create_trapezoidal_partitions()
        elif self.mf_type == MembershipFunctionType.GAUSSIAN:
            self.fuzzy_sets = self._create_gaussian_partitions()
        elif self.mf_type == MembershipFunctionType.BELL:
            self.fuzzy_sets = self._create_bell_partitions()
        else:
            raise ValueError(f"Unsupported membership function type: {self.mf_type}")
        
        return self.fuzzy_sets
    
    def _create_triangular_partitions(self) -> List[FuzzySet]:
        """
        Create triangular fuzzy set partitions with 50% overlap.
        
        The partitions are designed so that adjacent fuzzy sets overlap at the
        0.5 membership level, ensuring complete coverage of the universe.
        """
        fuzzy_sets = []
        n = self.num_partitions
        lower = self.universe.lower_bound
        upper = self.universe.upper_bound
        
        # Calculate the step size (distance between centers)
        step = (upper - lower) / (n - 1) if n > 1 else (upper - lower)
        
        for i in range(n):
            # Center of this fuzzy set
            center = lower + i * step
            
            # Left and right points of the triangle
            # For first set, left point is at the lower bound
            # For last set, right point is at the upper bound
            left = lower + (i - 1) * step if i > 0 else lower - step
            right = lower + (i + 1) * step if i < n - 1 else upper + step
            
            name = f"{self.prefix}{i + 1}"
            params = {'a': left, 'b': center, 'c': right}
            
            fuzzy_sets.append(FuzzySet(
                name=name,
                mf_type=MembershipFunctionType.TRIANGULAR,
                parameters=params,
                center=center
            ))
        
        return fuzzy_sets
    
    def _create_trapezoidal_partitions(self) -> List[FuzzySet]:
        """
        Create trapezoidal fuzzy set partitions.
        
        The first and last sets are half-trapezoids (extending to infinity),
        while middle sets are full trapezoids with overlapping regions.
        """
        fuzzy_sets = []
        n = self.num_partitions
        lower = self.universe.lower_bound
        upper = self.universe.upper_bound
        
        # Calculate width of each partition
        total_width = upper - lower
        # Each trapezoid has a flat top portion and sloped sides
        partition_width = total_width / n
        overlap = partition_width * 0.25  # 25% overlap on each side
        
        for i in range(n):
            center_left = lower + i * partition_width
            center_right = center_left + partition_width
            center = (center_left + center_right) / 2
            
            if i == 0:
                # First partition: left-open trapezoid
                a = lower - partition_width  # Extend beyond lower bound
                b = lower
                c = center_right - overlap
                d = center_right + overlap
            elif i == n - 1:
                # Last partition: right-open trapezoid
                a = center_left - overlap
                b = center_left + overlap
                c = upper
                d = upper + partition_width  # Extend beyond upper bound
            else:
                # Middle partitions: full trapezoid
                a = center_left - overlap
                b = center_left + overlap
                c = center_right - overlap
                d = center_right + overlap
            
            name = f"{self.prefix}{i + 1}"
            params = {'a': a, 'b': b, 'c': c, 'd': d}
            
            fuzzy_sets.append(FuzzySet(
                name=name,
                mf_type=MembershipFunctionType.TRAPEZOIDAL,
                parameters=params,
                center=center
            ))
        
        return fuzzy_sets
    
    def _create_gaussian_partitions(self) -> List[FuzzySet]:
        """
        Create Gaussian fuzzy set partitions.
        
        Centers are evenly spaced, and sigma is chosen so that adjacent
        Gaussians cross at approximately 0.5 membership.
        """
        fuzzy_sets = []
        n = self.num_partitions
        lower = self.universe.lower_bound
        upper = self.universe.upper_bound
        
        # Calculate step between centers
        step = (upper - lower) / (n - 1) if n > 1 else (upper - lower)
        
        # Sigma is chosen so that at distance step/2, membership is ~0.5
        # exp(-(step/2)^2 / (2*sigma^2)) = 0.5
        # sigma = step / (2 * sqrt(2 * ln(2)))
        sigma = step / (2 * np.sqrt(2 * np.log(2))) if n > 1 else (upper - lower) / 4
        
        for i in range(n):
            center = lower + i * step
            name = f"{self.prefix}{i + 1}"
            params = {'c': center, 'sigma': sigma}
            
            fuzzy_sets.append(FuzzySet(
                name=name,
                mf_type=MembershipFunctionType.GAUSSIAN,
                parameters=params,
                center=center
            ))
        
        return fuzzy_sets
    
    def _create_bell_partitions(self) -> List[FuzzySet]:
        """
        Create generalized bell-shaped fuzzy set partitions.
        
        The bell function provides more control over shape than Gaussian
        through the b parameter (slope steepness).
        """
        fuzzy_sets = []
        n = self.num_partitions
        lower = self.universe.lower_bound
        upper = self.universe.upper_bound
        
        # Calculate step between centers
        step = (upper - lower) / (n - 1) if n > 1 else (upper - lower)
        
        # Width parameter (a) controls where membership drops
        # Slope parameter (b) controls steepness, higher = steeper
        a = step / 2 if n > 1 else (upper - lower) / 2
        b = 2.0  # Moderate steepness
        
        for i in range(n):
            center = lower + i * step
            name = f"{self.prefix}{i + 1}"
            params = {'a': a, 'b': b, 'c': center}
            
            fuzzy_sets.append(FuzzySet(
                name=name,
                mf_type=MembershipFunctionType.BELL,
                parameters=params,
                center=center
            ))
        
        return fuzzy_sets


@dataclass
class FuzzyLogicalRelation:
    """
    Represents a single Fuzzy Logical Relation (FLR).
    
    For first-order: A_i -> A_j (single antecedent)
    For high-order: (A_i, A_j, ..., A_k) -> A_l (multiple antecedents)
    """
    antecedent: Tuple[str, ...]  # Tuple of fuzzy set names
    consequent: str  # Single fuzzy set name
    
    def __hash__(self):
        return hash((self.antecedent, self.consequent))
    
    def __eq__(self, other):
        if not isinstance(other, FuzzyLogicalRelation):
            return False
        return self.antecedent == other.antecedent and self.consequent == other.consequent
    
    def __str__(self):
        if len(self.antecedent) == 1:
            return f"{self.antecedent[0]} -> {self.consequent}"
        else:
            return f"({', '.join(self.antecedent)}) -> {self.consequent}"


@dataclass
class FuzzyLogicalRelationGroup:
    """
    Represents a Fuzzy Logical Relation Group (FLRG).
    
    Groups all consequents that share the same antecedent pattern.
    For example: A1 -> A2, A3, A4 (first-order)
    Or: (A1, A2) -> A3, A4 (high-order)
    """
    antecedent: Tuple[str, ...]
    consequents: List[str] = field(default_factory=list)
    
    def add_consequent(self, consequent: str):
        """Add a consequent to the group if not already present."""
        if consequent not in self.consequents:
            self.consequents.append(consequent)
    
    def __str__(self):
        if len(self.antecedent) == 1:
            ant_str = self.antecedent[0]
        else:
            ant_str = f"({', '.join(self.antecedent)})"
        
        if not self.consequents:
            return f"{ant_str} -> ∅"
        return f"{ant_str} -> {', '.join(self.consequents)}"


class FuzzyTimeSeries:
    """
    Main class for Fuzzy Time Series forecasting.
    
    Implements both First-Order FTS (FOFTS) and High-Order FTS (HOFTS)
    algorithms from scratch.
    """
    
    def __init__(
        self,
        order: int = 1,
        num_partitions: int = 7,
        mf_type: MembershipFunctionType = MembershipFunctionType.TRIANGULAR,
        margin_percent: float = 0.1
    ):
        """
        Initialize the Fuzzy Time Series model.
        
        Args:
            order: The order of the FTS model (1 for FOFTS, >1 for HOFTS)
            num_partitions: Number of fuzzy sets to create
            mf_type: Type of membership function to use
            margin_percent: Percentage margin to add to universe of discourse
        """
        self.order = order
        self.num_partitions = num_partitions
        self.mf_type = mf_type
        self.margin_percent = margin_percent
        
        # Will be set during training
        self.universe: Optional[UniverseOfDiscourse] = None
        self.fuzzy_sets: List[FuzzySet] = []
        self.fuzzy_set_map: Dict[str, FuzzySet] = {}
        self.flrs: List[FuzzyLogicalRelation] = []
        self.flrgs: Dict[Tuple[str, ...], FuzzyLogicalRelationGroup] = {}
        self.fuzzified_series: List[str] = []
        self.training_data: np.ndarray = np.array([])
        
    def fit(self, data: np.ndarray) -> 'FuzzyTimeSeries':
        """
        Fit the FTS model to training data.
        
        Steps:
        1. Define universe of discourse
        2. Create fuzzy set partitions
        3. Fuzzify the time series
        4. Generate FLRs and FLRGs
        
        Args:
            data: 1D numpy array of time series values
            
        Returns:
            self for method chaining
        """
        self.training_data = np.array(data).flatten()
        
        # Step 1: Define universe of discourse
        self._define_universe()
        
        # Step 2: Create fuzzy set partitions
        self._create_fuzzy_sets()
        
        # Step 3: Fuzzify the time series
        self._fuzzify_series()
        
        # Step 4: Generate FLRs
        self._generate_flrs()
        
        # Step 5: Generate FLRGs
        self._generate_flrgs()
        
        return self
    
    def _define_universe(self):
        """Define the universe of discourse based on training data."""
        min_val = np.min(self.training_data)
        max_val = np.max(self.training_data)
        
        # Calculate margin
        data_range = max_val - min_val
        margin = data_range * self.margin_percent
        
        self.universe = UniverseOfDiscourse(
            min_val=min_val,
            max_val=max_val,
            margin=margin
        )
    
    def _create_fuzzy_sets(self):
        """Create fuzzy set partitions using the configured partitioner."""
        partitioner = FuzzySetPartitioner(
            universe=self.universe,
            num_partitions=self.num_partitions,
            mf_type=self.mf_type
        )
        
        self.fuzzy_sets = partitioner.create_partitions()
        
        # Create mapping from name to fuzzy set for quick lookup
        self.fuzzy_set_map = {fs.name: fs for fs in self.fuzzy_sets}
    
    def _fuzzify_value(self, value: float) -> str:
        """
        Fuzzify a single crisp value by finding the fuzzy set with maximum membership.
        
        Args:
            value: Crisp input value
            
        Returns:
            Name of the fuzzy set with highest membership
        """
        max_membership = -1
        best_set = self.fuzzy_sets[0].name
        
        for fs in self.fuzzy_sets:
            membership = fs.membership(value)
            if membership > max_membership:
                max_membership = membership
                best_set = fs.name
        
        return best_set
    
    def _fuzzify_series(self):
        """Fuzzify all values in the training series."""
        self.fuzzified_series = [
            self._fuzzify_value(value) for value in self.training_data
        ]
    
    def _generate_flrs(self):
        """
        Generate Fuzzy Logical Relations based on the model order.
        
        For order k, each FLR is: (F(t-k), F(t-k+1), ..., F(t-1)) -> F(t)
        """
        self.flrs = []
        n = len(self.fuzzified_series)
        
        for t in range(self.order, n):
            # Get antecedent (previous k fuzzified values)
            antecedent = tuple(self.fuzzified_series[t - self.order:t])
            # Get consequent (current fuzzified value)
            consequent = self.fuzzified_series[t]
            
            flr = FuzzyLogicalRelation(antecedent=antecedent, consequent=consequent)
            self.flrs.append(flr)
    
    def _generate_flrgs(self):
        """
        Generate Fuzzy Logical Relation Groups by grouping FLRs with same antecedent.
        """
        self.flrgs = {}
        
        for flr in self.flrs:
            if flr.antecedent not in self.flrgs:
                self.flrgs[flr.antecedent] = FuzzyLogicalRelationGroup(
                    antecedent=flr.antecedent
                )
            self.flrgs[flr.antecedent].add_consequent(flr.consequent)
    
    def _defuzzify(self, fuzzy_set_names: List[str]) -> float:
        """
        Defuzzify by computing the average center of the consequent fuzzy sets.
        
        This implements the centroid defuzzification method.
        
        Args:
            fuzzy_set_names: List of fuzzy set names from the FLRG consequents
            
        Returns:
            Crisp output value
        """
        if not fuzzy_set_names:
            # If no consequents, return middle of universe
            return (self.universe.lower_bound + self.universe.upper_bound) / 2
        
        centers = [self.fuzzy_set_map[name].center for name in fuzzy_set_names]
        return np.mean(centers)
    
    def predict_next(self, history: List[float]) -> float:
        """
        Predict the next value given a history of recent values.
        
        Args:
            history: List of recent values (length should be >= order)
            
        Returns:
            Predicted next value
        """
        if len(history) < self.order:
            raise ValueError(f"History length ({len(history)}) must be >= order ({self.order})")
        
        # Take only the last 'order' values
        recent = history[-self.order:]
        
        # Fuzzify the recent values
        fuzzified = tuple(self._fuzzify_value(v) for v in recent)
        
        # Look up the FLRG
        if fuzzified in self.flrgs:
            consequents = self.flrgs[fuzzified].consequents
        else:
            # No matching rule - use nearest neighbor approach
            consequents = self._find_nearest_flrg(fuzzified)
        
        # Defuzzify
        return self._defuzzify(consequents)
    
    def _find_nearest_flrg(self, target: Tuple[str, ...]) -> List[str]:
        """
        Find the nearest FLRG when exact match is not found.
        
        Uses a simple similarity measure based on fuzzy set indices.
        
        Args:
            target: The target antecedent tuple
            
        Returns:
            List of consequent fuzzy set names from the nearest FLRG
        """
        if not self.flrgs:
            return [self.fuzzy_sets[len(self.fuzzy_sets) // 2].name]
        
        # Convert fuzzy set names to indices for distance calculation
        def get_index(name: str) -> int:
            for i, fs in enumerate(self.fuzzy_sets):
                if fs.name == name:
                    return i
            return 0
        
        target_indices = [get_index(name) for name in target]
        
        min_distance = float('inf')
        best_flrg = None
        
        for antecedent, flrg in self.flrgs.items():
            antecedent_indices = [get_index(name) for name in antecedent]
            
            # Calculate Manhattan distance
            distance = sum(abs(t - a) for t, a in zip(target_indices, antecedent_indices))
            
            if distance < min_distance:
                min_distance = distance
                best_flrg = flrg
        
        return best_flrg.consequents if best_flrg else [self.fuzzy_sets[0].name]
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Generate predictions for a series of data points.
        
        Args:
            data: Input time series data
            
        Returns:
            Array of predictions (first 'order' values cannot be predicted)
        """
        data = np.array(data).flatten()
        predictions = np.full(len(data), np.nan)
        
        # Start predicting from index 'order'
        for t in range(self.order, len(data)):
            history = list(data[:t])
            predictions[t] = self.predict_next(history)
        
        return predictions
    
    def forecast(self, steps: int, initial_history: List[float]) -> List[float]:
        """
        Forecast multiple steps into the future.
        
        Args:
            steps: Number of steps to forecast
            initial_history: Historical data to start from
            
        Returns:
            List of forecasted values
        """
        history = list(initial_history)
        forecasts = []
        
        for _ in range(steps):
            next_val = self.predict_next(history)
            forecasts.append(next_val)
            history.append(next_val)
        
        return forecasts
    
    def get_flrs_as_strings(self) -> List[str]:
        """Get all FLRs as human-readable strings."""
        return [str(flr) for flr in self.flrs]
    
    def get_flrgs_as_strings(self) -> List[str]:
        """Get all FLRGs as human-readable strings."""
        return [str(flrg) for flrg in self.flrgs.values()]
    
    def get_fuzzy_sets_info(self) -> List[Dict]:
        """Get information about all fuzzy sets."""
        return [
            {
                'name': fs.name,
                'type': fs.mf_type.value,
                'parameters': fs.parameters,
                'center': fs.center
            }
            for fs in self.fuzzy_sets
        ]


class FTSMetrics:
    """
    Calculates performance metrics for FTS models.
    """
    
    @staticmethod
    def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate Root Mean Square Error.
        
        RMSE = sqrt(mean((actual - predicted)^2))
        """
        # Remove NaN values
        mask = ~np.isnan(predicted) & ~np.isnan(actual)
        actual_clean = actual[mask]
        predicted_clean = predicted[mask]
        
        if len(actual_clean) == 0:
            return np.nan
        
        return np.sqrt(np.mean((actual_clean - predicted_clean) ** 2))
    
    @staticmethod
    def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate Mean Absolute Error.
        
        MAE = mean(|actual - predicted|)
        """
        mask = ~np.isnan(predicted) & ~np.isnan(actual)
        actual_clean = actual[mask]
        predicted_clean = predicted[mask]
        
        if len(actual_clean) == 0:
            return np.nan
        
        return np.mean(np.abs(actual_clean - predicted_clean))
    
    @staticmethod
    def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate Mean Absolute Percentage Error.
        
        MAPE = mean(|actual - predicted| / |actual|) * 100
        
        Note: Excludes zero actual values to avoid division by zero.
        """
        mask = ~np.isnan(predicted) & ~np.isnan(actual) & (actual != 0)
        actual_clean = actual[mask]
        predicted_clean = predicted[mask]
        
        if len(actual_clean) == 0:
            return np.nan
        
        return np.mean(np.abs((actual_clean - predicted_clean) / actual_clean)) * 100
    
    @staticmethod
    def all_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        """Calculate all metrics at once."""
        return {
            'RMSE': FTSMetrics.rmse(actual, predicted),
            'MAE': FTSMetrics.mae(actual, predicted),
            'MAPE': FTSMetrics.mape(actual, predicted)
        }
