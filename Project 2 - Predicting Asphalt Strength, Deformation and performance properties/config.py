import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Asphalt-Dataset-ToClass.xlsx")

TEST_RATIO = 0.20
RANDOM_SEED = 42

# Column definitions (matching the Excel layout)
INPUT_COLUMNS = [
    "Viscosity",       # Pa·s
    "Pb",              # % A.C. by weight
    "Pbe",             # % Effective Asphalt Content
    "Gmm",             # Maximum Theoretical Specific Gravity
    "Va",              # % Air Voids
    "UnitWeight",      # kg/m³
    "P200",            # % Passing 0.075 mm
    "P4",              # Cumulative Percent Retained 4.75 mm
    "P38",             # Cumulative Percent Retained 9.5 mm
    "P34",             # Cumulative Percent Retained 19 mm
]

OUTPUT_COLUMNS = [
    "Stability",       # Adjusted Stability (kN)
    "Flow",            # Flow (mm)
    "ITSM20",          # ITSM at 20°C (MPa)
    "ITSM30",          # ITSM at 30°C (MPa)
]

ALL_COLUMNS = INPUT_COLUMNS + OUTPUT_COLUMNS

MF_TYPE = "gaussian"

TSK_ORDER = 1

# Subtractive clustering parameters
CLUSTER_RADIUS = 1.2          # ra - neighbourhood radius (normalised space)
SQUASH_FACTOR = 1.25          # ratio for rb = squash_factor * ra
ACCEPT_RATIO = 0.5            # threshold for accepting a cluster center
REJECT_RATIO = 0.15           # threshold for rejecting a cluster center

# Optimisation parameters
LEARNING_RATE = 0.01
MAX_EPOCHS = 800
TOLERANCE = 1e-8
