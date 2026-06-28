"""
Run the full ACB-UQ benchmark.

For each dataset and each random split we record, at target coverage 1-alpha:
  coverage, mean interval width, worst-slice (conditional) coverage, runtime.
Results are aggregated (mean +/- std) over SEEDS splits and written to
results/main_results.csv. A separate adaptive-sampling study is written to
results/adaptive_curve.csv.
"""

import os
import time
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from acbuq.datasets import LOADERS
from acbuq import methods as M

ALPHA = 0.10                      # 90% target intervals
SEEDS = list(range(10))
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)


def split_data(X, y, seed):
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.5,
                                            random_state=seed)
    Xcal, Xte, ycal, yte = train_test_split(Xtmp, ytmp, test_size=0.5,
                                            random_state=seed)
    return Xtr, ytr, Xcal, ycal, Xte, yte


def eval_interval(name, X, y, lo, hi, runtime):
    return {
        "method": name,
        "coverage": M.empirical_coverage(y, lo, hi),
        "width": M.mean_width(lo, hi),
        "worst_slice": M.worst_slice_coverage(X, y, lo, hi),
        "runtime_s": runtime,
    }


def run_main():
    rows = []
    for dname, loader in LOADERS.items():
        X, y, _ = loader()
        for seed in SEEDS:
            Xtr, ytr, Xcal, ycal, Xte, yte = split_data(X, y, seed)
            yrange = float(np.max(y) - np.min(y))

            def timed(fn, *a, **k):
                t = time.perf_counter()
                out = fn(*a, **k)
                return out, time.perf_counter() - t

            # ACB-UQ (ours)
            (lo, hi, cert), rt = timed(M.acb_uq, Xtr, ytr, Xcal, ycal,
                                       Xte, yte, ALPHA, seed)
            r = eval_interval("ACB-UQ (ours)", Xte, yte, lo, hi, rt)
            r.update({"dataset": dname, "seed": seed,
                      "width_norm": r["width"] / yrange,
                      "cred_lo": cert["cred_lo"], "cred_hi": cert["cred_hi"],
                      "p_meets": cert["p_meets_target"]})
            rows.append(r)

            # CQR (adaptive, no Bayesian layer)
            (lohi), rt = timed(M.cqr, Xtr, ytr, Xcal, ycal, Xte, ALPHA, seed)
            lo, hi = lohi
            r = eval_interval("CQR", Xte, yte, lo, hi, rt)
            r.update({"dataset": dname, "seed": seed,
                      "width_norm": r["width"] / yrange})
            rows.append(r)

            # Split conformal
            (out), rt = timed(M.split_conformal, Xtr, ytr, Xcal, ycal,
                              Xte, ALPHA, seed)
            lo, hi, _ = out
            r = eval_interval("Split-Conformal", Xte, yte, lo, hi, rt)
            r.update({"dataset": dname, "seed": seed,
                      "width_norm": r["width"] / yrange})
            rows.append(r)

            # Deep ensemble
            (out), rt = timed(M.deep_ensemble, Xtr, ytr, Xcal, ycal,
                              Xte, ALPHA, 5, seed)
            lo, hi = out
            r = eval_interval("Deep-Ensemble", Xte, yte, lo, hi, rt)
            r.update({"dataset": dname, "seed": seed,
                      "width_norm": r["width"] / yrange})
            rows.append(r)

            # Single MLP Gaussian
            (out), rt = timed(M.single_mlp_gauss, Xtr, ytr, Xcal, ycal,
                              Xte, ALPHA, seed)
            lo, hi = out
            r = eval_interval("MLP-Gaussian", Xte, yte, lo, hi, rt)
            r.update({"dataset": dname, "seed": seed,
                      "width_norm": r["width"] / yrange})
            rows.append(r)

            # Gaussian process
            (out), rt = timed(M.gaussian_process, Xtr, ytr, Xcal, ycal,
                              Xte, ALPHA, seed)
            lo, hi = out
            r = eval_interval("Gaussian-Process", Xte, yte, lo, hi, rt)
            r.update({"dataset": dname, "seed": seed,
                      "width_norm": r["width"] / yrange})
            rows.append(r)

            # Bayesian ridge
            (out), rt = timed(M.bayesian_ridge, Xtr, ytr, Xcal, ycal,
                              Xte, ALPHA, seed)
            lo, hi = out
            r = eval_interval("Bayesian-Ridge", Xte, yte, lo, hi, rt)
            r.update({"dataset": dname, "seed": seed,
                      "width_norm": r["width"] / yrange})
            rows.append(r)
        print(f"[done] {dname}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "main_results_raw.csv"), index=False)

    agg = (df.groupby(["dataset", "method"])
             .agg(coverage_mean=("coverage", "mean"),
                  coverage_std=("coverage", "std"),
                  width_norm_mean=("width_norm", "mean"),
                  width_norm_std=("width_norm", "std"),
                  worst_slice_mean=("worst_slice", "mean"),
                  runtime_mean=("runtime_s", "mean"))
             .reset_index())
    agg.to_csv(os.path.join(RES, "main_results.csv"), index=False)
    return df, agg


def run_adaptive(n_seeds=15):
    """Active high-uncertainty sampling, averaged over seeds. Start from a small
    calibration set and iteratively add the pool points with the widest
    predicted interval (adaptive) versus uniformly random points (random),
    re-fitting the adaptive CQR intervals and tracking width and coverage."""
    all_curves = []
    for seed in range(n_seeds):
        X, y, _ = LOADERS["friedman1"](n_samples=3000, seed=100 + seed)
        Xtr, Xpool, ytr, ypool = train_test_split(X, y, test_size=0.7,
                                                  random_state=seed)
        Xpool, Xte, ypool, yte = train_test_split(Xpool, ypool, test_size=0.5,
                                                  random_state=seed)
        rng = np.random.default_rng(seed)

        for strategy in ("adaptive", "random"):
            cal_idx = list(rng.choice(len(Xpool), 50, replace=False))
            remaining = set(range(len(Xpool))) - set(cal_idx)
            for _ in range(8):
                Xcal, ycal = Xpool[cal_idx], ypool[cal_idx]
                lo, hi = M.cqr(Xtr, ytr, Xcal, ycal, Xte, ALPHA, seed=seed)
                all_curves.append({"strategy": strategy, "seed": seed,
                                   "budget": len(cal_idx),
                                   "coverage": M.empirical_coverage(yte, lo, hi),
                                   "width": M.mean_width(lo, hi)})
                rem = np.array(sorted(remaining))
                if strategy == "adaptive":
                    plo, phi = M.cqr(Xtr, ytr, Xcal, ycal, Xpool[rem],
                                     ALPHA, seed=seed)
                    add = rem[np.argsort(-(phi - plo))[:30]]
                else:
                    add = rng.choice(rem, 30, replace=False)
                for a in add:
                    cal_idx.append(int(a)); remaining.discard(int(a))
    df = pd.DataFrame(all_curves)
    agg = (df.groupby(["strategy", "budget"])
             .agg(coverage=("coverage", "mean"),
                  width=("width", "mean"),
                  width_std=("width", "std"))
             .reset_index())
    agg.to_csv(os.path.join(RES, "adaptive_curve.csv"), index=False)
    print("[done] adaptive study")
    return agg


if __name__ == "__main__":
    df, agg = run_main()
    run_adaptive()
    pd.set_option("display.width", 200)
    print(agg.to_string(index=False))
