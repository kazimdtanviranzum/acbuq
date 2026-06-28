"""
Dataset loaders for the ACB-UQ experiments.

Three regression tasks are provided:

1. diabetes        : REAL data (sklearn.datasets.load_diabetes). 442 patients,
                     10 standardized clinical features, target = disease
                     progression one year after baseline. Stands in for the
                     "medical decision support" intelligent system.

2. friedman1       : SIMULATED. The canonical Friedman-1 nonlinear regression
                     benchmark (Friedman, 1991), widely used in the conformal /
                     uncertainty-quantification literature. Heteroscedastic
                     Gaussian noise is added so that interval *adaptivity* can be
                     measured. Stands in for a high-dimensional black-box surrogate.

3. degradation     : SIMULATED. A physics-based exponential bearing-degradation /
                     remaining-useful-life (RUL) model with measurement noise that
                     grows with degradation (heteroscedastic). Stands in for the
                     "industrial condition-monitoring" intelligent system.

All simulators are fully seeded and therefore reproducible.
"""

import numpy as np
from sklearn.datasets import load_diabetes


def get_diabetes():
    d = load_diabetes()
    X = np.asarray(d.data, dtype=float)
    y = np.asarray(d.target, dtype=float)
    return X, y, "diabetes"


def get_friedman1(n_samples=2000, n_features=10, noise_base=0.5, seed=0):
    """Friedman-1 function with heteroscedastic noise.

    y = 10 sin(pi x1 x2) + 20 (x3 - 0.5)^2 + 10 x4 + 5 x5 + eps,
    x_j ~ U(0,1); only the first 5 features are informative.
    Noise std grows linearly with x4 to create non-constant uncertainty.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=(n_samples, n_features))
    signal = (10.0 * np.sin(np.pi * X[:, 0] * X[:, 1])
              + 20.0 * (X[:, 2] - 0.5) ** 2
              + 10.0 * X[:, 3]
              + 5.0 * X[:, 4])
    sigma = noise_base * (1.0 + 2.0 * X[:, 3])          # heteroscedastic
    y = signal + rng.normal(0.0, sigma)
    return X, y, "friedman1"


def get_degradation(n_units=400, seed=1):
    """Physics-based bearing-degradation / RUL simulator.

    Each unit i has a latent exponential degradation path
        h_i(t) = h0 * exp(beta_i * t) + measurement noise,
    with unit-specific rate beta_i. We observe a feature vector summarising the
    most recent monitoring window (current health index, short-term slope,
    rolling RMS energy, kurtosis proxy, operating load, ambient temperature,
    cumulative running time, plus 5 correlated nuisance sensors) and regress the
    remaining useful life (RUL, in cycles) to a fixed failure threshold.

    Measurement noise increases with degradation, producing genuine
    heteroscedastic uncertainty that an adaptive interval should capture.
    """
    rng = np.random.default_rng(seed)
    h0 = 0.05
    fail = 1.0                                          # failure threshold
    feats, rul = [], []
    for _ in range(n_units):
        beta = rng.uniform(0.015, 0.06)                # degradation rate
        load = rng.uniform(0.5, 1.5)
        temp = rng.uniform(20.0, 80.0)
        beta_eff = beta * (0.7 + 0.6 * load) * (0.8 + 0.4 * (temp / 80.0))
        t_fail = np.log(fail / h0) / beta_eff
        # observe at a random point in the first 80% of life
        t_obs = rng.uniform(0.0, 0.8 * t_fail)
        h = h0 * np.exp(beta_eff * t_obs)
        meas_noise = rng.normal(0.0, 0.02 + 0.15 * h)  # grows with health index
        h_meas = max(h + meas_noise, 1e-3)
        slope = beta_eff * h_meas                       # local derivative
        rms = h_meas * (1.0 + 0.1 * rng.standard_normal())
        kurt = 3.0 + 6.0 * h_meas + 0.3 * rng.standard_normal()
        run_time = t_obs
        nuisance = h_meas * rng.uniform(0.5, 1.5, size=5) + rng.normal(0, 0.05, 5)
        x = np.concatenate([[h_meas, slope, rms, kurt, load, temp, run_time],
                            nuisance])
        feats.append(x)
        rul.append(t_fail - t_obs)
    X = np.asarray(feats, dtype=float)
    y = np.asarray(rul, dtype=float)
    return X, y, "degradation"


LOADERS = {
    "diabetes": get_diabetes,
    "friedman1": get_friedman1,
    "degradation": get_degradation,
}
