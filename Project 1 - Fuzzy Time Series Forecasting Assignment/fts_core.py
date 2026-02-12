import numpy as np
from enum import Enum
from collections import defaultdict


class MembershipFunctionType(Enum):
    TRIANGULAR = "triangular"
    TRAPEZOIDAL = "trapezoidal"
    GAUSSIAN = "gaussian"
    BELL = "bell"


class FuzzySet:
    def __init__(self, name, mf_type, parameters, center):
        self.name = name
        self.mf_type = mf_type
        self.parameters = parameters
        self.center = center

    def membership(self, x):
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

    def _triangular_membership(self, x):
        """
              1 |     /\
                |    /  \
                |   /    \
              0 |__/______\__
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

    def _trapezoidal_membership(self, x):
        """
              1 |    ____
                |   /    \
                |  /      \
              0 |_/________\__
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

    def _gaussian_membership(self, x):
        c = self.parameters['c']
        sigma = self.parameters['sigma']

        return np.exp(-((x - c) ** 2) / (2 * sigma ** 2))

    def _bell_membership(self, x):
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

    def __init__(self, min_val, max_val, margin=0.0):
        self.min_val = min_val
        self.max_val = max_val
        self.margin = margin

    def lower_bound(self):
        return self.min_val - self.margin

    def upper_bound(self):
        return self.max_val + self.margin

    def get_range(self):
        return self.upper_bound() - self.lower_bound()


class FuzzySetPartitioner:
    def __init__(self, universe, num_partitions, mf_type=MembershipFunctionType.TRIANGULAR, prefix="A"):
        self.universe = universe
        self.num_partitions = num_partitions
        self.mf_type = mf_type
        self.prefix = prefix
        self.fuzzy_sets = []

    def create_partitions(self):
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

    def _create_triangular_partitions(self):
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

    def _create_trapezoidal_partitions(self):
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
            Instead, we "push" its sides outward so adjacent sets overlap.

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

    def _create_gaussian_partitions(self):
        fuzzy_sets = []
        n = self.num_partitions
        lower = self.universe.lower_bound()
        upper = self.universe.upper_bound()

        width = (upper - lower) / n

        # Typically, we want Gaussians to overlap at μ ≈ 0.5 (between adjacent centers).
        # This happens if sigma ~ width / (2 * sqrt(2 * ln(2))) ≈ 0.425 * width.
        sigma = width / (2 * np.sqrt(2 * np.log(2)))

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

    def _create_bell_partitions(self):
        fuzzy_sets = []
        n = self.num_partitions
        lower = self.universe.lower_bound()
        upper = self.universe.upper_bound()

        width = (upper - lower) / n

        # Bell parameters: similar to Gaussian, but use the generalized bell shape
        # a (width) ~ half-width of each partition
        # b (slope) ~ controls steepness; b=2 is a common choice
        # c (center) ~ center point of each partition
        a = width * 0.6
        b = 2

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
    def __init__(self, antecedent, consequent):
        self.antecedent = antecedent
        self.consequent = consequent

    def __str__(self):
        if isinstance(self.antecedent, tuple):
            antecedent_str = ", ".join(str(a) for a in self.antecedent)
        else:
            antecedent_str = str(self.antecedent)
        return f"({antecedent_str}) -> {self.consequent}"


class FuzzyLogicalRelationGroup:
    def __init__(self, antecedent):
        self.antecedent = antecedent
        self.consequents = []

    def add_consequent(self, consequent):
        if consequent not in self.consequents:
            self.consequents.append(consequent)

    def __str__(self):
        if isinstance(self.antecedent, tuple):
            antecedent_str = ", ".join(str(a) for a in self.antecedent)
        else:
            antecedent_str = str(self.antecedent)

        consequent_str = ", ".join(self.consequents)
        return f"({antecedent_str}) -> {consequent_str}"


class FuzzyTimeSeries:

    def __init__(self, order=1, num_partitions=7, mf_type=MembershipFunctionType.TRIANGULAR, margin_percent=0.1):
        self.order = order
        self.num_partitions = num_partitions
        self.mf_type = mf_type
        self.margin_percent = margin_percent

        self.universe = None
        self.fuzzy_sets = []
        self.fuzzy_set_map = {}
        self.training_data = None
        self.fuzzified_series = []
        self.flrs = []
        self.flrgs = {}

    def fit(self, data):
        """
        Train the fuzzy time series model on the given data.

        Steps:
        1. Define universe of discourse
        2. Create fuzzy sets (partitions)
        3. Fuzzify the time series
        4. Generate FLRs (Fuzzy Logical Relations)
        5. Group FLRs into FLRGs
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

    def _fuzzify_value(self, value):
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

    def _defuzzify(self, fuzzy_set_names):
        if not fuzzy_set_names:
            # If no consequents -> return middle of universe
            return (self.universe.lower_bound() + self.universe.upper_bound()) / 2

        centers = [self.fuzzy_set_map[name].center for name in fuzzy_set_names]
        return np.mean(centers)

    def predict_next(self, history):
        if len(history) < self.order:
            return

        recent = history[-self.order:]
        fuzzified = tuple(self._fuzzify_value(v) for v in recent)

        if fuzzified in self.flrgs:
            consequents = self.flrgs[fuzzified].consequents
        else:
            consequents = self._find_nearest_flrg(fuzzified)

        return self._defuzzify(consequents)

    def _find_nearest_flrg(self, target):
        """Find the nearest FLRG when exact match is not found."""
        def get_index(name):
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

    def predict(self, data):
        # Return a copy of the array collapsed into one dimension.
        # Read More: https://numpy.org/devdocs/reference/generated/numpy.ndarray.flatten.html
        data = np.array(data).flatten()
        predictions = np.full(len(data), np.nan)

        for t in range(self.order, len(data)):
            history = list(data[:t])
            predictions[t] = self.predict_next(history)

        return predictions

    def forecast(self, steps, initial_history):
        """This function predicts multiple steps into the unknown future by using its own predictions as input."""
        history = list(initial_history)
        forecasts = []

        for _ in range(steps):
            next_val = self.predict_next(history)
            forecasts.append(next_val)
            history.append(next_val)

        return forecasts

    def get_flrs_as_strings(self):
        """Human-readable FLRs."""
        return [str(flr) for flr in self.flrs]

    def get_flrgs_as_strings(self):
        """Human-readable FLRGs."""
        return [str(flrg) for flrg in self.flrgs.values()]

    def get_fuzzy_sets_info(self):
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
    def rmse(self, actual, predicted):
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

    def mae(self, actual, predicted):
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

    def mape(self, actual, predicted):
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

    def all_metrics(self, actual, predicted):
        return {
            'RMSE': self.rmse(actual, predicted),
            'MAE': self.mae(actual, predicted),
            'MAPE': self.mape(actual, predicted)
        }
