# Predictive Maintenance of Turbofan Engines via Multichannel Deep Learning and RUL Estimation

Master's Thesis (MSc in Artificial Intelligence, VIU, 2026). Design and rigorous comparison of three architectures for predicting the **Remaining Useful Life (RUL)** of turbofan engines, with epistemic and aleatoric **uncertainty quantification**, evaluated on NASA's C-MAPSS benchmark.

## Overview

Unplanned maintenance of turbofan engines costs up to €10M per event and is responsible for roughly 30% of commercial aviation delays. This project implements and compares three Prognostics and Health Management (PHM) approaches, a gradient-boosted tree ensemble and two deep learning architectures, to predict RUL from 21 simultaneous sensor channels, producing uncertainty estimates suitable for real maintenance decisions rather than point estimates alone.

## Dataset

[NASA C-MAPSS](https://www.nasa.gov/intelligent-systems-division/) (Saxena et al., 2008): simulated turbofan engine degradation trajectories, split into 4 subsets of increasing complexity (1 or 6 operating conditions, 1 or 2 fault modes), each with 100 to 249 engines.

## Models Implemented

- **XGBoost**: gradient-boosted trees using per-sensor window statistics (mean, std, min, max, range over a 60-cycle window) plus shared degradation features (cumulative delta, local slope, acceleration).
- **CNN + BiLSTM + Bahdanau Attention**: hybrid architecture combining convolutional feature extraction with a bidirectional LSTM and a temporal attention mechanism that focuses on the most informative time steps.
- **Temporal Fusion Transformer (TFT)**: self-attention architecture adapted from multi-horizon forecasting to RUL regression, with operating condition integrated as a static covariate.

## Key Methodological Contribution

Published CMAPSS literature commonly reports metrics inflated by data leakage, global train/test splits and scalers fit before splitting the data. This project instead uses:

- **Engine-level train/validation split** (`GroupShuffleSplit`), no engine appears in both sets.
- **Data normalization per operational condition via K-Means clustering**, applied separately per subset, instead of a single global scaler that lets the operating regime mask the actual degradation signal.

This produces a leakage-free evaluation protocol.

## Uncertainty Quantification

Each model estimates uncertainty, using a different technique appropriate to its architecture:

| Model | Technique |
|---|---|
| TFT | Quantile regression (multi-quantile output heads) (aleatoric) |
| CNN-BiLSTM | Monte Carlo Dropout (epistemic) + heteroscedastic output head (aleatoric) |
| XGBoost | Quantile regression (aleatoric) |

## Results

RMSE and NASA Score across the four CMAPSS subsets:

| Subset | RMSE — CNN | RMSE — TFT | RMSE — XGB | Score — CNN | Score — TFT | Score — XGB |
|---|---|---|---|---|---|---|
| FD001 | 15.40 | **14.76** | 15.50 | 385.2 | **373** | 535 |
| FD002 | 22.20 | 24.31 | **14.56** | 7,022.8 | 9,860 | **762** |
| FD003 | 21.01 | 18.35 | **16.79** | 3,321.9 | **930** | 610 |
| FD004 | 23.39 | 32.05 | **22.41** | 5,768.9 | 55,997 | **4,043** |

**Findings:**
- XGBoost achieved the best overall RMSE/Score on 3 of 4 subsets, outperforming both deep learning architectures by up to 10 RMSE points on the multi-condition subsets (FD002, FD004). Its gradient boosting architecture grants this model better performance with fewer data compared to deep learning models.
- TFT stood out on FD001 (the single, simplest subset), where multi-head attention captures long-range temporal dependencies without interference from operating-regime variability.
- TFT's quantile output produced the tightest uncertainty bands (24–35 cycles at [p10, p90]) vs. XGBoost's 60–85 cycles.
- Despite the stricter, leakage-free protocol (not directly comparable to most published CMAPSS baselines) results remain competitive with the published state of the art.

## Repository Structure

```
.
├── CMAPSSDataset.py      # Shared preprocessing: degradation features, RUL labeling, K-Means per-condition normalization
├── XGBoost.ipynb         # XGBoost model, quantile regression, ablation study
├── TorchCNNBiLSTM.ipynb  # CNN-BiLSTM-Attention (PyTorch), MC Dropout + heteroscedastic head
└── TFT.ipynb             # Temporal Fusion Transformer, quantile regression
```

Trained model weights are not tracked in this repository, every result is reproducible by running the notebooks against the CMAPSS dataset.

## Setup

```bash
git clone https://github.com/mfernl/MPMT_RUL_TFM.git
cd MPMT_RUL_TFM
pip install -r requirements.txt
jupyter notebook
```

Download the NASA C-MAPSS dataset (Saxena et al., 2008) and point the paths in `CMAPSSDataset.py` to your local copy.

## Tech Stack

Python · PyTorch · XGBoost · scikit-learn · pandas · NumPy · SciPy · Matplotlib

## Author

Marco Fernández Llamas — [github.com/mfernl](https://github.com/mfernl)
