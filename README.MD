
![Fuzzy sets and Logic](https://res.cloudinary.com/dyjew8iji/image/upload/v1771064627/Fuzzy_sets_and_Logic_aj0dvl.webp)

# Fuzzy-Sets-and-Fuzzy-Logic

A comprehensive repository implementing Fuzzy Logic Systems for time series forecasting and regression problems. This repository contains two major projects demonstrating practical applications of fuzzy set theory in real-world scenarios.

## 📋 Repository Overview

This repository explores **Fuzzy Set Theory** and **Fuzzy Logic Systems** through two distinct implementations:

1. **Fuzzy Time Series Forecasting** - Predicting temporal patterns using First-Order and High-Order Fuzzy Time Series models
2. **TSK Fuzzy System for Asphalt Property Prediction** - Predicting mechanical and performance properties of asphalt mixtures

All implementations are built **from scratch** without relying on pre-existing fuzzy logic libraries, providing deep insights into the underlying algorithms and methodologies.

---

## 🎯 Project 1: Fuzzy Time Series Forecasting

### Objective
Implement and analyze Fuzzy Time Series (FTS) models using both **First-Order FTS (FOFTS)** and **High-Order FTS (HOFTS)** for forecasting applications.

### Datasets
- **Mackey-Glass Time Series**: A chaotic, non-linear benchmark dataset
- **Influenza Cases Data**: Forecasting specimens detected with Influenza Type A and Type B

### Key Features
- Custom fuzzification algorithms with multiple membership function types (triangular, trapezoidal, Gaussian)
- Fuzzy Logical Rule Groups (FLRGs) generation
- Parameter tuning for order and partition numbers
- Performance evaluation using RMSE, MAE, and MAPE metrics
- Interactive interface for n-step ahead forecasting

### Implementation Components
1. Data preprocessing and train-test splitting (80/20)
2. Universe of discourse definition
3. Fuzzy set creation with chosen membership functions
4. Fuzzification and defuzzification processes
5. FLRG generation for first-order and high-order models
6. Forecasting engine with performance metrics

---

## 🏗️ Project 2: Asphalt Property Prediction (TSK Fuzzy System)

### Objective
Design a **Takagi-Sugeno-Kang (TSK) Fuzzy System** to predict critical asphalt pavement properties for enhanced road safety and infrastructure durability.

### Problem Domain
Accurate prediction of asphalt properties is crucial for:
- Safety of motorists, bikers, and pedestrians
- Resource allocation and scheduling in road construction
- Quality control and project success
- Long-term infrastructure durability

### Input Features
- **Binder Properties**: Viscosity (η)
- **Asphalt Content**: % A.C. by weight (Pb), % Effective Asphalt Content (Pbe)
- **Density & Volumetric**: Maximum Theoretical Specific Gravity (Gmm), Unit Weight, % Air Voids (Va)
- **Aggregate Gradation**: P200, P4, P38, P34 (particle size distribution)

### Output Predictions
1. **Adjusted Stability** (kN) - Load-carrying capacity
2. **Flow** (mm) - Deformation characteristics
3. **ITSM @ 20°C** - Indirect Tensile Stiffness Modulus at moderate temperature
4. **ITSM @ 30°C** - Stiffness at high temperature (rutting resistance)

### Key Features
- TSK fuzzy rule generation and tuning
- Feature selection/engineering methods
- Separate or unified fuzzy systems for multi-output prediction
- Comprehensive performance evaluation using RMSE on train (80%) and test (20%) data

---

## 📊 Evaluation Metrics

### Project 1 (FTS)
- **RMSE** (Root Mean Square Error)
- **MAE** (Mean Absolute Error)
- **MAPE** (Mean Absolute Percentage Error)

### Project 2 (TSK)
- **RMSE** calculated separately for:
  - Training data (80%)
  - Testing data (20%)
  - Each output variable (Flow, Stability, ITSM20, ITSM30)

---

## 🚀 Getting Started

### Prerequisites
- Implementation language and dependencies are specified in each project directory
- No pre-built fuzzy logic libraries are used - all algorithms implemented from scratch

---

## 📈 Deliverables

Each project includes:
- ✅ Complete source code with detailed comments
- ✅ Comprehensive reports documenting methodology and results
- ✅ Visualizations (time series plots, error metrics, membership functions)
- ✅ Performance analysis and comparison tables
- ✅ User interface for predictions

---

## 📝 Reports

All reports include:
- Detailed methodology descriptions
- Fuzzy set and rule generation processes
- Membership function design and tuning approaches
- Results and performance analysis
- Code execution and usage instructions

---

## 🎓 Academic Context

This repository is part of the **Fuzzy Sets and Systems** coursework for Fall 2025, demonstrating practical applications of fuzzy logic theory in:
- Time series forecasting with linguistic uncertainty
- Regression problems with complex input-output relationships
- Real-world engineering applications

---

## ⚠️ Important Notes

- All fuzzy logic algorithms are implemented **from scratch**
- No pre-existing fuzzy time series or fuzzy system libraries are used

---
