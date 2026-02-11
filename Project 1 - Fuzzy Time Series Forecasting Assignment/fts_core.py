import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Callable
from enum import Enum
from collections import defaultdict


class MembershipFunctionType(Enum):
    TRIANGULAR = "triangular"
    TRAPEZOIDAL = "trapezoidal"
    GAUSSIAN = "gaussian"
    BELL = "bell"


class FuzzySet:
    def __init__(self, name: str, mf_type: MembershipFunctionType, parameters: Dict[str, float], center: float):
        self.name = name
        self.mf_type = mf_type
        self.parameters = parameters
        self.center = center

    def membership(self, x: float) -> float:
        if self.mf_type == MembershipFunctionType.TRIANGULAR:
            return self._triangular_membership(x)
        elif self.mf_type == MembershipFunctionType.TRAPEZOIDAL:
            return self._trapezoidal_membership(x)
        elif self.mf_type == MembershipFunctionType.GAUSSIAN:
            return self._gaussian_membership(x)
        elif self.mf_type == MembershipFunctionType.BELL:
            return self._bell_membership(x)
        else:
            raise ValueError(
                f"Unknown membership function type: {self.mf_type}")

    def _triangular_membership(self, x: float) -> float:
        """
              1 |     /\\
                |    /  \\
                |   /    \\
              0 |__/______\\__
                  a   b   c
        """
        a = self.parameters['a']
        b = self.parameters['b']
        c = self.parameters['c']

        if x <= a or x >= c:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a) if b != a else 1.0
        else:
            return (c - x) / (c - b) if c != b else 1.0

    def _trapezoidal_membership(self, x: float) -> float:
        """
              1 |    ____
                |   /    \\
                |  /      \\
              0 |_/________\\__
                 a   b   c  d
        """
        a = self.parameters['a']
        b = self.parameters['b']
        c = self.parameters['c']
        d = self.parameters['d']

        if x <= a or x >= d:
            return 0.0
        elif a < x < b:
            return (x - a) / (b - a) if b != a else 1.0
        elif b <= x <= c:
            return 1.0
        else:
            return (d - x) / (d - c) if d != c else 1.0

    def _gaussian_membership(self, x: float) -> float:
        c = self.parameters['c']
        sigma = self.parameters['sigma']

        return np.exp(-((x - c) ** 2) / (2 * sigma ** 2))

    def _bell_membership(self, x: float) -> float:
        """
        a (width), b (slope), c (center)
        μ(x) = 1 / (1 + |((x - c) / a)|^(2b))
        """
        a = self.parameters['a']  # Width
        b = self.parameters['b']  # Slope
        c = self.parameters['c']  # Center

        if a == 0:
            return 1.0 if x == c else 0.0

        return 1 / (1 + np.abs((x - c) / a) ** (2 * b))


class UniverseOfDiscourse:

    def __init__(self, min_val: float, max_val: float, margin: float = 0.0):
        self.min_val = min_val
        self.max_val = max_val
        self.margin = margin

    def lower_bound(self) -> float:
        return self.min_val - self.margin

    def upper_bound(self) -> float:
        return self.max_val + self.margin

    def get_range(self) -> float:
        return self.upper_bound() - self.lower_bound()


class FuzzySetPartitioner:
    def __init__(
        self,
        universe: UniverseOfDiscourse,
        num_partitions: int,
        mf_type: MembershipFunctionType = MembershipFunctionType.TRIANGULAR,
        prefix: str = "A"
    ):
        self.universe = universe
        self.num_partitions = num_partitions
        self.mf_type = mf_type
        self.prefix = prefix
        self.fuzzy_sets: List[FuzzySet] = []

    def create_partitions(self) -> List[FuzzySet]:
        if self.mf_type == MembershipFunctionType.TRIANGULAR:
            self.fuzzy_sets = self._create_triangular_partitions()
        elif self.mf_type == MembershipFunctionType.TRAPEZOIDAL:
            self.fuzzy_sets = self._create_trapezoidal_partitions()
        elif self.mf_type == MembershipFunctionType.GAUSSIAN:
            self.fuzzy_sets = self._create_gaussian_partitions()
        elif self.mf_type == MembershipFunctionType.BELL:
            self.fuzzy_sets = self._create_bell_partitions()
        else:
            raise ValueError(
                f"Unsupported membership function type: {self.mf_type}")

        return self.fuzzy_sets

    def _create_triangular_partitions(self) -> List[FuzzySet]:
        fuzzy_sets = []
        n = self.num_partitions
        lower = self.universe.lower_bound()
        upper = self.universe.upper_bound()

        width = (upper - lower) / n if n > 0 else (upper - lower)

        for i in range(n):
            cl = lower + i * width
            cr = cl + width

            center = (cl + cr) / 2

            # Expand into neighbors so triangles overlap smoothly
            # (crossing near μ = 0.5 instead of touching at one point)
            left = cl - width / 2
            right = cr + width / 2

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
        """ The first and last sets are half-trapezoids (extending to infinity),
            while middle sets are full trapezoids with overlapping regions.
        """

        """
            1) Split the universe into n equal base segments:
                width = (U - L) / n

            Each fuzzy set i corresponds to one base segment:
                cl = L + i * width        # left edge of the segment
                cr = cl + width           # right edge of the segment
                center = (cl + cr) / 2    # geometric center of the fuzzy set


            2) Add overlap so neighboring sets blend smoothly.
            Without overlap, trapezoids would touch at one point and create
            sharp jumps. We therefore *expand* each trapezoid into its neighbors
            by 25% of the segment width:

                overlap = 0.25 * width

            Interpretation:
                subtract overlap -> extend into the previous set
                add overlap      -> extend into the next set


            3) Build a trapezoid (a, b, c, d) for each segment:

                a -> b : rising edge  (membership grows from 0 to 1)
                b -> c : flat top     (membership = 1)
                c -> d : falling edge (membership falls from 1 to 0)


            Why the + and −:
            ----------------
            We do NOT want the trapezoid to start and end exactly at cl and cr.
            Instead, we “push” its sides outward so adjacent sets overlap.

                a = cl - overlap   # start rising BEFORE the core segment
                b = cl + overlap   # reach full membership slightly INSIDE it
                c = cr - overlap   # start falling BEFORE leaving the core
                d = cr + overlap   # finish falling AFTER leaving the core
        """
        fuzzy_sets = []
        n = self.num_partitions
        lower = self.universe.lower_bound()
        upper = self.universe.upper_bound()

        total_width = upper - lower
        partition_width = total_width / n
        overlap = partition_width * 0.25  # 25% overlap on each side

        for i in range(n):
            center_left = lower + i * partition_width
            center_right = center_left + partition_width
            center = (center_left + center_right) / 2
            # right-open trapezoid (Left Shoulder)
            if i == 0:
                a = lower - partition_width
                b = lower
                c = center_right - overlap
                d = center_right + overlap
            # right-open trapezoid (Right Shoulder)
            elif i == n - 1:
                a = center_left - overlap
                b = center_left + overlap
                c = upper
                d = upper + partition_width
            # Middle partitions: full trapezoid
            else:
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
        Sigma is chosen so that neighboring Gaussians intersect
        at μ = 0.5 (half-membership).
        """
        fuzzy_sets = []
        n = self.num_partitions
        lower = self.universe.lower_bound()
        upper = self.universe.upper_bound()

        width = (upper - lower) / n if n > 0 else (upper - lower)

        step = width

        # Set the mean condition to 0.5 at x = step/2
        # exp(-(step/2)^2 / (2*sigma^2)) = 0.5

        # Step 1: Rearranging the equation
        # Taking the natural logarithm of both sides
        # - (step/2)^2 / (2*sigma^2) = ln(0.5)

        # Step 2: ln(0.5) can be rewritten using properties of logarithms
        # - (step/2)^2 / (2*sigma^2) = -ln(2)

        # Step 3: Remove negative signs
        # (step/2)^2 / (2*sigma^2) = ln(2)

        # Step 4: Multiplying both sides by 2*sigma^2
        # (step/2)^2 = 2*sigma^2 * ln(2)

        # Step 5: Isolating sigma^2
        # sigma^2 = (step^2 / 4) / (2*ln(2))

        # Step 6: Taking the square root to solve for sigma
        # sigma = (step / 2) / sqrt(2*ln(2))
        sigma = step / (2 * np.sqrt(2 * np.log(2))
                        ) if n > 0 else (upper - lower) / 4

        for i in range(n):
            cl = lower + i * width
            cr = cl + width
            center = (cl + cr) / 2

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
            All bells are expanded so that neighboring bells intersect at
            μ = 0.5, ensuring smooth fuzzy transitions.

            The generalized bell MF is:
                μ(x) = 1 / (1 + |(x - c) / a|^(2b))

            where:
                c = center of the bell
                a = half-width control (where μ ≈ 0.5)
                b = slope/steepness (higher = sharper sides)
        """
        fuzzy_sets = []
        n = self.num_partitions
        lower = self.universe.lower_bound()
        upper = self.universe.upper_bound()

        width = (upper - lower) / n if n > 0 else (upper - lower)

        step = width

        # Choose a so that μ(step/2) = 0.5:
        #   1 / (1 + |(step/2)/a|^(2b)) = 0.5
        # -> |(step/2)/a|^(2b) = 1
        # -> (step/2)/a = 1
        # -> a = step/2
        a = step / 2 if n > 0 else (upper - lower) / 2

        b = 2.0

        for i in range(n):
            cl = lower + i * width
            cr = cl + width
            center = (cl + cr) / 2

            name = f"{self.prefix}{i + 1}"
            params = {'a': a, 'b': b, 'c': center}

            fuzzy_sets.append(FuzzySet(
                name=name,
                mf_type=MembershipFunctionType.BELL,
                parameters=params,
                center=center
            ))

        return fuzzy_sets


class FuzzyLogicalRelation:
    def __init__(self, antecedent: Tuple[str, ...], consequent: str):
        self.antecedent = antecedent
        self.consequent = consequent

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


class FuzzyLogicalRelationGroup:
    def __init__(self, antecedent: Tuple[str, ...], consequents: List[str] = None):
        self.antecedent = antecedent
        self.consequents = consequents if consequents is not None else []

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
            return f"{ant_str} -> {{}}"
        return f"{ant_str} -> {', '.join(self.consequents)}"


class FuzzyTimeSeries:
    """
    Main class for Fuzzy Time Series forecasting.
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
        """
        # Return a copy of the array collapsed into one dimension.
        # Read More: https://numpy.org/devdocs/reference/generated/numpy.ndarray.flatten.html
        self.training_data = np.array(data).flatten()

        self._define_universe()
        self._create_fuzzy_sets()
        self._fuzzify_series()
        self._generate_flrs()
        self._generate_flrgs()

        return self

    def _define_universe(self):
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
        partitioner = FuzzySetPartitioner(
            universe=self.universe,
            num_partitions=self.num_partitions,
            mf_type=self.mf_type
        )

        self.fuzzy_sets = partitioner.create_partitions()

        # Create dict for quick lookup
        self.fuzzy_set_map = {fs.name: fs for fs in self.fuzzy_sets}

    def _fuzzify_value(self, value: float) -> str:
        """
            1. Start with the first category as "best guess"
            2. Check each category to see how well the value fits
            3. Keep the category with the highest "fit score"
            4. Return that category name
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
        # Converts every number in your training data to its fuzzy category.
        # If training_data = [3.2, 7.5, 9.1]
        # Result might be: ["Low", "Medium", "High"]
        self.fuzzified_series = [
            self._fuzzify_value(value) for value in self.training_data
        ]

    def _generate_flrs(self):
        self.flrs = []
        n = len(self.fuzzified_series)

        for t in range(self.order, n):
            antecedent = tuple(self.fuzzified_series[t - self.order:t])
            consequent = self.fuzzified_series[t]

            flr = FuzzyLogicalRelation(
                antecedent=antecedent, consequent=consequent)
            self.flrs.append(flr)

    def _generate_flrgs(self):
        self.flrgs = {}

        for flr in self.flrs:
            if flr.antecedent not in self.flrgs:
                self.flrgs[flr.antecedent] = FuzzyLogicalRelationGroup(
                    antecedent=flr.antecedent
                )
            self.flrgs[flr.antecedent].add_consequent(flr.consequent)

    def _defuzzify(self, fuzzy_set_names: List[str]) -> float:
        if not fuzzy_set_names:
            # If no consequents -> return middle of universe
            return (self.universe.lower_bound() + self.universe.upper_bound()) / 2

        centers = [self.fuzzy_set_map[name].center for name in fuzzy_set_names]
        return np.mean(centers)

    def predict_next(self, history: List[float]) -> float:
        if len(history) < self.order:
            return

        recent = history[-self.order:]
        fuzzified = tuple(self._fuzzify_value(v) for v in recent)

        if fuzzified in self.flrgs:
            consequents = self.flrgs[fuzzified].consequents
        else:
            consequents = self._find_nearest_flrg(fuzzified)

        return self._defuzzify(consequents)

    def _find_nearest_flrg(self, target: Tuple[str, ...]) -> List[str]:
        """Find the nearest FLRG when exact match is not found."""
        def get_index(name: str) -> int:
            for i, fs in enumerate(self.fuzzy_sets):
                if fs.name == name:
                    return i
            return 0
        # If no rules exist at all, return the middle fuzzy set
        if not self.flrgs:
            return [self.fuzzy_sets[len(self.fuzzy_sets) // 2].name]

        best_flrg = None
        min_distance = float('inf')
        target_indices = [get_index(name) for name in target]

        for antecedent, flrg in self.flrgs.items():
            antecedent_indices = [get_index(name) for name in antecedent]

            # Calculate Manhattan distance
            distance = sum(abs(t - a)
                           for t, a in zip(target_indices, antecedent_indices))

            if distance < min_distance:
                min_distance = distance
                best_flrg = flrg

        return best_flrg.consequents if best_flrg else [self.fuzzy_sets[0].name]

    def predict(self, data: np.ndarray) -> np.ndarray:
        # Return a copy of the array collapsed into one dimension.
        # Read More: https://numpy.org/devdocs/reference/generated/numpy.ndarray.flatten.html
        data = np.array(data).flatten()
        predictions = np.full(len(data), np.nan)

        for t in range(self.order, len(data)):
            history = list(data[:t])
            predictions[t] = self.predict_next(history)

        return predictions

    def forecast(self, steps: int, initial_history: List[float]) -> List[float]:
        """This function predicts multiple steps into the unknown future by using its own predictions as input."""
        history = list(initial_history)
        forecasts = []

        for _ in range(steps):
            next_val = self.predict_next(history)
            forecasts.append(next_val)
            history.append(next_val)

        return forecasts

    def get_flrs_as_strings(self) -> List[str]:
        """Human-readable FLRs."""
        return [str(flr) for flr in self.flrs]

    def get_flrgs_as_strings(self) -> List[str]:
        """Human-readable FLRGs."""
        return [str(flrg) for flrg in self.flrgs.values()]

    def get_fuzzy_sets_info(self) -> List[Dict]:
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
    def rmse(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Root Mean Square Error:

        RMSE = sqrt(mean((actual - predicted)^2))
        """
        mask = ~np.isnan(predicted) & ~np.isnan(actual)
        actual_clean = actual[mask]
        predicted_clean = predicted[mask]

        if len(actual_clean) == 0:
            return np.nan

        return np.sqrt(np.mean((actual_clean - predicted_clean) ** 2))

    def mae(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Mean Absolute Error:

        MAE = mean(|actual - predicted|)
        """
        mask = ~np.isnan(predicted) & ~np.isnan(actual)
        actual_clean = actual[mask]
        predicted_clean = predicted[mask]

        if len(actual_clean) == 0:
            return np.nan

        return np.mean(np.abs(actual_clean - predicted_clean))

    def mape(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Mean Absolute Percentage Error:

        MAPE = mean(|actual - predicted| / |actual|) * 100

        ****Note*****: Excludes zero actual values to avoid division by zero.
        """
        mask = ~np.isnan(predicted) & ~np.isnan(actual) & (actual != 0)
        actual_clean = actual[mask]
        predicted_clean = predicted[mask]

        if len(actual_clean) == 0:
            return np.nan

        return np.mean(np.abs((actual_clean - predicted_clean) / actual_clean)) * 100

    def all_metrics(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        return {
            'RMSE': self.rmse(actual, predicted),
            'MAE': self.mae(actual, predicted),
            'MAPE': self.mape(actual, predicted)
        }
