"""
Core methods for Adaptive Conformal-Bayesian Uncertainty Quantification (ACB-UQ).

The module implements, all from first principles on top of scikit-learn:

  * a feed-forward neural-network surrogate of the mean (MLPRegressor);
  * split conformal regression (Lei et al., 2018);
  * conformalized quantile regression / CQR (Romano et al., 2019), the adaptive
    core of ACB-UQ, using gradient-boosted quantile surrogates;
  * a Bayesian Beta-Binomial certification layer that returns a posterior
    credible interval on the achieved coverage and a confidence-calibrated
    effective level alpha_eff (the ACB contribution);
  * baselines: deep ensemble (Gaussian), single-MLP Gaussian, Gaussian process,
    Bayesian ridge.

Every estimator below is real and runs without a network connection.
"""

import time
import numpy as np
from scipy import stats
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def conformal_quantile_level(n_cal, alpha):
    """Finite-sample conformal level: ceil((n+1)(1-alpha)) / n."""
    return min(1.0, np.ceil((n_cal + 1) * (1.0 - alpha)) / n_cal)


def empirical_coverage(y, lo, hi):
    return float(np.mean((y >= lo) & (y <= hi)))


def mean_width(lo, hi):
    return float(np.mean(hi - lo))


def worst_slice_coverage(X, y, lo, hi, n_bins=5):
    """Worst per-feature-slice coverage: a proxy for conditional coverage.

    For each feature we bin the test points into n_bins quantile slices and
    record the minimum coverage across all slices and features.
    """
    covered = (y >= lo) & (y <= hi)
    worst = 1.0
    for j in range(X.shape[1]):
        edges = np.quantile(X[:, j], np.linspace(0, 1, n_bins + 1))
        idx = np.clip(np.digitize(X[:, j], edges[1:-1]), 0, n_bins - 1)
        for b in range(n_bins):
            m = idx == b
            if m.sum() >= 10:
                worst = min(worst, float(np.mean(covered[m])))
    return worst


# --------------------------------------------------------------------------- #
# Bayesian Beta-Binomial certification (the ACB layer)
# --------------------------------------------------------------------------- #
def beta_binomial_coverage(n_covered, n_total, alpha_target,
                           prior_strength=20.0, cred=0.90):
    """Posterior on the true coverage probability p.

    Prior  : Beta(a0, b0) centred at the target 1-alpha with the given strength.
    Likelih: Binomial(n_total, p) with n_covered successes.
    Returns (posterior_mean, lower_cred, upper_cred, P[p >= 1-alpha_target]).
    """
    target = 1.0 - alpha_target
    a0 = prior_strength * target
    b0 = prior_strength * (1.0 - target)
    a = a0 + n_covered
    b = b0 + (n_total - n_covered)
    post_mean = a / (a + b)
    lo = stats.beta.ppf((1 - cred) / 2, a, b)
    hi = stats.beta.ppf(1 - (1 - cred) / 2, a, b)
    p_meets = 1.0 - stats.beta.cdf(target, a, b)
    return post_mean, lo, hi, p_meets


def certified_alpha(n_cal, alpha_target, prior_strength=20.0, tau=0.90):
    """Choose an effective miscoverage level so that, under the Beta-Binomial
    posterior over the calibration miscoverage rate, the probability of meeting
    the nominal 1-alpha_target coverage on future data is at least tau.

    This is a Bayesian, training-conditional analogue of PAC conformal
    prediction. It shrinks alpha slightly when the calibration set is small.
    """
    target = 1.0 - alpha_target
    a0 = prior_strength * target
    b0 = prior_strength * (1.0 - target)
    # We want the LARGEST alpha_eff <= alpha_target (i.e. the narrowest, least
    # conservative intervals) such that the tau-lower-credible bound on the
    # resulting coverage still meets the nominal target. Scan from large to
    # small and return the first level that satisfies the certificate.
    grid = np.linspace(alpha_target, alpha_target * 0.3, 80)
    best = alpha_target * 0.3
    for a_eff in grid:
        exp_cov = 1.0 - a_eff
        a = a0 + exp_cov * n_cal
        b = b0 + (1.0 - exp_cov) * n_cal
        q_low = stats.beta.ppf(1 - tau, a, b)
        if q_low >= target:
            best = a_eff
            break
    return float(best)


# --------------------------------------------------------------------------- #
# Surrogate (mean) + split conformal
# --------------------------------------------------------------------------- #
def make_mlp(seed=0):
    return MLPRegressor(hidden_layer_sizes=(64, 64), activation="relu",
                        alpha=1e-3, max_iter=2000, early_stopping=True,
                        n_iter_no_change=20, random_state=seed)


def split_conformal(Xtr, ytr, Xcal, ycal, Xte, alpha, seed=0):
    sc = StandardScaler().fit(Xtr)
    model = make_mlp(seed).fit(sc.transform(Xtr), ytr)
    res = np.abs(ycal - model.predict(sc.transform(Xcal)))
    q = np.quantile(res, conformal_quantile_level(len(ycal), alpha),
                    method="higher")
    pred = model.predict(sc.transform(Xte))
    return pred - q, pred + q, pred


# --------------------------------------------------------------------------- #
# Conformalized Quantile Regression (adaptive core)
# --------------------------------------------------------------------------- #
def fit_quantile_pair(Xtr, ytr, alpha, seed=0):
    lo = GradientBoostingRegressor(loss="quantile", alpha=alpha / 2,
                                   n_estimators=300, max_depth=3,
                                   learning_rate=0.05, subsample=0.8,
                                   random_state=seed).fit(Xtr, ytr)
    hi = GradientBoostingRegressor(loss="quantile", alpha=1 - alpha / 2,
                                   n_estimators=300, max_depth=3,
                                   learning_rate=0.05, subsample=0.8,
                                   random_state=seed).fit(Xtr, ytr)
    return lo, hi


def cqr(Xtr, ytr, Xcal, ycal, Xte, alpha, seed=0, alpha_fit=None):
    """Conformalized quantile regression.

    alpha_fit controls the nominal quantiles of the underlying GBR pair; the
    conformal correction is computed at level alpha. Passing alpha_fit = alpha
    recovers standard CQR.
    """
    if alpha_fit is None:
        alpha_fit = alpha
    qlo, qhi = fit_quantile_pair(Xtr, ytr, alpha_fit, seed)
    lo_cal, hi_cal = qlo.predict(Xcal), qhi.predict(Xcal)
    E = np.maximum(lo_cal - ycal, ycal - hi_cal)
    Q = np.quantile(E, conformal_quantile_level(len(ycal), alpha),
                    method="higher")
    lo_te, hi_te = qlo.predict(Xte), qhi.predict(Xte)
    return lo_te - Q, hi_te + Q


def acb_uq(Xtr, ytr, Xcal, ycal, Xte, yte, alpha, seed=0,
           prior_strength=20.0, tau=0.90):
    """Full ACB-UQ predictor.

    1. adaptive CQR intervals with a Bayesian-certified effective level;
    2. Beta-Binomial posterior certification of the achieved coverage.
    Returns intervals plus a certification dict.
    """
    a_eff = certified_alpha(len(ycal), alpha, prior_strength, tau)
    lo, hi = cqr(Xtr, ytr, Xcal, ycal, Xte, a_eff, seed=seed, alpha_fit=alpha)
    n_cov = int(np.sum((yte >= lo) & (yte <= hi)))
    pm, cl, cu, pmeet = beta_binomial_coverage(n_cov, len(yte), alpha,
                                               prior_strength)
    cert = {"alpha_eff": a_eff, "post_mean_cov": pm,
            "cred_lo": cl, "cred_hi": cu, "p_meets_target": pmeet}
    return lo, hi, cert


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
def deep_ensemble(Xtr, ytr, Xcal, ycal, Xte, alpha, n_models=5, seed=0):
    sc = StandardScaler().fit(Xtr)
    preds = []
    for k in range(n_models):
        m = make_mlp(seed + k).fit(sc.transform(Xtr), ytr)
        preds.append(m.predict(sc.transform(Xte)))
    preds = np.vstack(preds)
    mu = preds.mean(0)
    sd = preds.std(0) + 1e-6
    z = stats.norm.ppf(1 - alpha / 2)
    return mu - z * sd, mu + z * sd


def single_mlp_gauss(Xtr, ytr, Xcal, ycal, Xte, alpha, seed=0):
    sc = StandardScaler().fit(Xtr)
    m = make_mlp(seed).fit(sc.transform(Xtr), ytr)
    sd = np.std(ycal - m.predict(sc.transform(Xcal))) + 1e-6
    z = stats.norm.ppf(1 - alpha / 2)
    pred = m.predict(sc.transform(Xte))
    return pred - z * sd, pred + z * sd


def gaussian_process(Xtr, ytr, Xcal, ycal, Xte, alpha, seed=0, cap=600):
    sc = StandardScaler().fit(Xtr)
    if len(Xtr) > cap:                                  # keep GP tractable
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(Xtr), cap, replace=False)
        Xtr, ytr = Xtr[idx], ytr[idx]
    kernel = (ConstantKernel(1.0) * RBF(length_scale=1.0)
              + WhiteKernel(noise_level=1.0))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                  random_state=seed).fit(sc.transform(Xtr), ytr)
    mu, sd = gp.predict(sc.transform(Xte), return_std=True)
    z = stats.norm.ppf(1 - alpha / 2)
    return mu - z * sd, mu + z * sd


def bayesian_ridge(Xtr, ytr, Xcal, ycal, Xte, alpha, seed=0):
    sc = StandardScaler().fit(Xtr)
    m = BayesianRidge().fit(sc.transform(Xtr), ytr)
    mu, sd = m.predict(sc.transform(Xte), return_std=True)
    z = stats.norm.ppf(1 - alpha / 2)
    return mu - z * sd, mu + z * sd
