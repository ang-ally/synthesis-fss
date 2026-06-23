# Machine Learning-Based Design Parameter Prediction for Square Loop Frequency Selective Surfaces (FSS)

## Overview

This project presents a Machine Learning-based approach for synthesizing the design parameters of a Square Loop Frequency Selective Surface (FSS) on an FR-4 substrate. The objective is to predict the geometric dimensions of the FSS structure from desired frequency specifications, eliminating the need for repetitive electromagnetic simulations and reducing design time.

The project utilizes Support Vector Regression (SVR) models to predict:

* **Track Length (d)**
* **Track Width (s)**

A Streamlit-based web application was developed to provide a simple and interactive interface for real-time FSS synthesis.

---

## Problem Statement

Designing Frequency Selective Surfaces typically involves multiple iterations of electromagnetic simulation to determine suitable geometric dimensions that satisfy desired frequency characteristics.

This project aims to automate the synthesis process by using Machine Learning models trained on simulated FSS datasets.

---

## Dataset

The dataset consists of **324 Square Loop FSS samples** generated through electromagnetic simulations.

### Input Features

| Feature | Description                   |
| ------- | ----------------------------- |
| h       | Substrate Height (mm)         |
| fr      | Resonant Frequency (GHz)      |
| fl      | Lower Cutoff Frequency (GHz)  |
| fh      | Higher Cutoff Frequency (GHz) |
| BW      | Bandwidth (GHz)               |
| FBW     | Fractional Bandwidth          |
| g       | Inter-element Spacing (mm)    |

### Output Parameters

| Output | Description       |
| ------ | ----------------- |
| d      | Track Length (mm) |
| s      | Track Width (mm)  |

---

## Machine Learning Models

### Model 1: Track Length Prediction (d)

**Algorithm:** Support Vector Regression (SVR)

**Hyperparameters**

```python
SVR(
    kernel='rbf',
    C=7000,
    gamma=0.1,
    epsilon=0.01
)
```

### Performance

| Metric   | Value     |
| -------- | --------- |
| R² Score | 0.9963    |
| RMSE     | 0.9663 mm |
| MAE      | 0.6320 mm |

---

### Model 2: Track Width Prediction (s)

**Algorithm:** Support Vector Regression (SVR)

**Preprocessing:** RobustScaler

**Hyperparameters**

```python
SVR(
    kernel='rbf',
    C=7000,
    gamma=0.1,
    epsilon=0.01
)
```

### Performance

| Metric   | Value     |
| -------- | --------- |
| R² Score | 0.9008    |
| RMSE     | 0.0869 mm |
| MAE      | 0.0537 mm |

---

## Project Workflow

```text
Frequency Specifications
        │
        ▼
Feature Extraction
(h, fr, fl, fh, BW, FBW, g)
        │
        ▼
SVR Model 1 ─────► Track Length (d)

SVR Model 2 ─────► Track Width (s)
        │
        ▼
Streamlit Web Application
        │
        ▼
Synthesized FSS Design Parameters
```

---

## Streamlit Application

The trained models were deployed using Streamlit to create a user-friendly synthesis tool.

### User Inputs

* Substrate Height (h)
* Resonant Frequency (fr)
* Lower Cutoff Frequency (fl)
* Higher Cutoff Frequency (fh)
* Inter-element Spacing (g)

### Calculated Parameters

* Bandwidth (BW)
* Fractional Bandwidth (FBW)

### Outputs

* Predicted Track Length (d)
* Predicted Track Width (s)

### Validation Rules

The application validates:

```text
fl < fr < fh
```

and provides appropriate error messages for invalid inputs.

---

## Example Prediction

### Input

```python
h  = 1.6
fr = 2.44
fl = 1.6187
fh = 3.4761
g  = 0.25
```

### Output

```text
Predicted Track Length (d): 15.47 mm

Predicted Track Width (s): 0.97 mm
```

---

## Technologies Used

* Python
* NumPy
* Pandas
* Scikit-Learn
* Streamlit
* Pickle

---

## Model Deployment

The trained models are serialized using Pickle.

```python
pickle.dump(model1, open('d_prediction.sav', 'wb'))
pickle.dump(model2, open('s_prediction.sav', 'wb'))
```

The saved models are loaded in the Streamlit application for inference.

---

## Applications

* Frequency Selective Surface Design
* RF and Microwave Engineering
* Electromagnetic Structure Synthesis
* Rapid Design Space Exploration
* Antenna and Filter Engineering

---

## Future Enhancements

* Support for multiple substrate materials.
* Multi-output regression model for simultaneous prediction.
* Integration with electromagnetic simulators for automated validation.
* Extension to different FSS geometries.
* Cloud deployment for online access.

---

## Author

**Angelyn Sweety I**

Electronics and Communication Engineer
Embedded Software Engineer | Machine Learning Enthusiast

Final Year Project – St. Joseph's College of Engineering
