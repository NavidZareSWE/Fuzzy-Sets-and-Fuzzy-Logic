# Asphalt TSK Fuzzy Prediction System

Predicts asphalt strength, deformation, and performance properties using
first-order Takagi–Sugeno–Kang (TSK) fuzzy inference systems.

## Outputs Predicted
- Adjusted Stability (kN)
- Flow (mm)
- ITSM at 20 °C (MPa)
- ITSM at 30 °C (MPa)

## Installation
```bash
pip install -r requirements.txt
```

## Training
```bash
python main.py
```

## Prediction (end-user)
```bash
python predict.py
```

## Project Structure
```
├── config.py               # All tuneable parameters
├── data_loader.py           # Data I/O and normalisation
├── membership_functions.py  # Gaussian MFs
├── clustering.py            # Subtractive clustering
├── tsk_system.py            # TSK rules and inference engine
├── training.py              # Hybrid learning (LSE + gradient descent)
├── evaluation.py            # RMSE computation
├── main.py                  # Training pipeline
├── predict.py               # Interactive prediction interface
├── data/                    # Dataset (Excel)
└── output/                  # Trained models and results
```
