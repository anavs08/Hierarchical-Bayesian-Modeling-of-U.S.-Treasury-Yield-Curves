# Hierarchical Bayesian Modeling of U.S. Treasury Yield-Curve Factors

This repository contains code + a short paper exploring **how different information-sharing (pooling) structures affect Bayesian time-series modeling** of the classic **Nelson–Siegel yield curve factors**: **Level, Slope, Curvature**.

We compare:
- **No pooling** (estimate each factor independently)
- **Partial pooling (hierarchical Bayes)** (share strength across factors via group-level priors)

The modeling goal is **stable estimation and forecasting** across regimes (e.g., crisis periods / rate-hike cycles), where independent fits can become high-variance and fully pooled fits can become biased.

---

## What’s in this repo

- `Final_Proj.ipynb`
  - End-to-end notebook that:
    - loads U.S. Treasury yield curve factor data (BETA0–BETA2),
    - maps to **Level/Slope/Curvature**,
    - fits Bayesian AR(1)-style dynamics with **heavy-tailed Student-t observations** and **stochastic volatility**,
    - compares **no pooling vs hierarchical partial pooling**.

- `Hierarchical_Bayesian_Modeling_Of_U_S__Treasury_Rate.pdf`
  - Paper writeup describing the probabilistic framework, pooling architectures, dataset, inference (NUTS/HMC), and empirical results. :contentReference[oaicite:0]{index=0}

---

## Modeling overview (high level)

### Factors
We work with Nelson–Siegel latent factors:
- **Level (BETA0)**: long-run rate level
- **Slope (BETA1)**: short-vs-long spread / policy cycle signal
- **Curvature (BETA2)**: medium-term hump / uncertainty

### Time-series dynamics
Each factor is modeled with AR(1)-type persistence, but with a **robust likelihood**:
- **Student-t** innovations to handle fat tails
- **Stochastic volatility** to allow time-varying noise

### Pooling structures
- **No pooling:** each factor has its own parameters (flexible, can overfit)
- **Partial pooling:** factor parameters are drawn from **shared hyperpriors**, inducing shrinkage/regularization and improving stability

---

## Data

The notebook expects a CSV with the **GSW-style** yield-curve factor columns:
- `BETA0`, `BETA1`, `BETA2`

In the notebook, this is read as something like:
- `/content/sample_data/feds200628.csv` (data has been attached as well)

If you run locally, update the file path to wherever you store the CSV.

---

## Requirements

The notebook uses:
- Python 3.9+
- `pandas`, `numpy`, `matplotlib`
- `pymc`, `pytensor`, `arviz`
- (optional) `jax` for acceleration if available

Install:
```bash
pip install pandas numpy matplotlib pymc pytensor arviz jax jaxlib
