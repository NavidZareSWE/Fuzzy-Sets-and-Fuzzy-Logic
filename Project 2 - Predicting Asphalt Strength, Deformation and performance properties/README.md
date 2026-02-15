# Asphalt TSK Fuzzy Prediction System

Predicts asphalt strength, deformation, and performance properties using
first-order Takagi-Sugeno-Kang (TSK) fuzzy inference systems.

## Input Variables (10)
- Viscosity (Pa·s)
- Pb (% A.C. by weight)
- Pbe (% Effective Asphalt Content)
- Gmm (Maximum Theoretical Specific Gravity)
- Va (% Air Voids)
- UnitWeight (kg/m³)
- P200 (% Passing 0.075 mm)
- P4 (Cumulative Percent Retained 4.75 mm)
- P38 (Cumulative Percent Retained 9.5 mm)
- P34 (Cumulative Percent Retained 19 mm)

## Output Variables (4)
- Stability - Adjusted Stability (kN)
- Flow - Flow (mm)
- ITSM20 - ITSM at 20°C (MPa)
- ITSM30 - ITSM at 30°C (MPa)

## Installation
```bash
pip install -r requirements.txt
```

## Training
```bash
python main.py
```

This will:
1. Load and preprocess the dataset
2. Build and tune TSK fuzzy systems for each output
3. Evaluate RMSE on training and test sets
4. Generate plots and save results to output/

## Prediction (End User)
```bash
python predict.py
```

This allows interactive prediction by entering 10 input values.

## Project Structure
```
├── config.py               # Configuration parameters
├── data_loader.py          # Data I/O and normalisation
├── membership_functions.py # Gaussian membership functions
├── clustering.py           # Subtractive clustering
├── tsk_system.py           # TSK rules and inference engine
├── training.py             # Hybrid learning (LSE + gradient descent)
├── main.py                 # Training pipeline
├── predict.py              # Interactive prediction interface
├── data/                   # Dataset (Excel)
└── output/                 # Trained models, results, and plots
```

## Methodology

### 1. Data Preprocessing
- 80/20 train-test split (random seed = 42)
- Min-max normalisation to [0, 1]

### 2. Rule Generation (Subtractive Clustering)
- Clustering performed on joint input-output space
- Cluster centers become rule prototypes
- Parameters: ra=1.2, squash_factor=1.25, accept_ratio=0.5, reject_ratio=0.15

### 3. Fuzzy System Structure
- First-order TSK (linear consequents)
- Gaussian membership functions
- Product t-norm for rule firing strength
- Weighted average defuzzification

### 4. Parameter Tuning (Hybrid Learning)
- Consequent parameters: Least Squares Estimation (LSE)
- Antecedent parameters: Gradient descent
- 800 epochs, learning rate = 0.01
