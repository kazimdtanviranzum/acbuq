"""Generate all manuscript figures (colorful) from the real result files."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy import stats

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11,
                     "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 300})

DISPLAY = {"Split-Conformal":"Split conformal","Deep-Ensemble":"Deep ensemble","MLP-Gaussian":"MLP gaussian","Gaussian-Process":"Gaussian process","Bayesian-Ridge":"Bayesian ridge","ACBUQ (ours)":"ACBUQ (ours)","CQR":"CQR"}
DSORDER = ["diabetes", "friedman1", "degradation"]
DSLABEL = {"diabetes": "Diabetes (real)", "friedman1": "Friedman 1 (sim.)",
           "degradation": "Degradation (sim.)"}
PALETTE = {"ACBUQ (ours)": "#d1495b", "CQR": "#00798c",
           "Split-Conformal": "#66a182", "Deep-Ensemble": "#edae49",
           "MLP-Gaussian": "#8d96a3", "Gaussian-Process": "#2e4057",
           "Bayesian-Ridge": "#b388eb"}
ORDER = ["ACBUQ (ours)", "CQR", "Split-Conformal", "Gaussian-Process",
         "Bayesian-Ridge", "MLP-Gaussian", "Deep-Ensemble"]


# --------------------------------------------------------------------------- #
def fig1_architecture():
    fig, ax = plt.subplots(figsize=(11, 5.0))
    ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 6)

    def box(x, y, w, h, text, fc, tc="white"):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.12",
                           linewidth=1.5, edgecolor="#222222", facecolor=fc, alpha=0.95)
        ax.add_patch(b)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=10.5, color=tc, weight="bold", wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                     mutation_scale=18, linewidth=1.8, color="#333333"))

    box(0.2, 2.4, 1.9, 1.2, "Intelligent\nsystem f(x)\n(black box)", "#2e4057")
    box(0.2, 4.2, 1.9, 1.0, "Operational\ndata", "#8d96a3")
    box(2.7, 3.1, 2.2, 1.4, "NN surrogate\n$\\hat{f}_\\theta(x)$\n(mean MLP)", "#00798c")
    box(2.7, 1.2, 2.2, 1.4, "Quantile pair\n$\\hat q_{lo},\\hat q_{hi}$\n(GBR)", "#66a182")
    box(5.5, 2.2, 2.3, 1.5, "Adaptive\nconformal\ncalibration (CQR)", "#d1495b")
    box(8.4, 3.3, 2.0, 1.4, "Bayesian\nBeta Binomial\ncertification", "#b388eb")
    box(8.4, 1.2, 2.0, 1.5, "Certified\ninterval\n$[\\hat L(x),\\hat U(x)]$\n+ coverage cert.", "#edae49", tc="#222")

    arrow(2.1, 3.0, 2.7, 3.4)
    arrow(2.1, 4.6, 3.4, 4.05)
    arrow(2.1, 2.9, 2.7, 2.0)
    arrow(4.9, 3.6, 5.5, 3.2)
    arrow(4.9, 1.9, 5.5, 2.5)
    arrow(7.8, 3.2, 8.4, 3.8)
    arrow(7.8, 2.7, 8.4, 2.2)
    arrow(9.4, 3.3, 9.4, 2.7)
    ax.text(6.0, 5.4, "ACBUQ: Adaptive Conformal Bayesian Uncertainty Quantification",
            ha="center", fontsize=13, weight="bold", color="#222")
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_architecture.png"), bbox_inches="tight")
    plt.close(fig)


def fig2_coverage_width():
    df = pd.read_csv(os.path.join(RES, "main_results.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    x = np.arange(len(DSORDER)); w = 0.115
    for i, m in enumerate(ORDER):
        sub = df[df.method == m].set_index("dataset")
        cov = [sub.loc[d, "coverage_mean"] for d in DSORDER]
        err = [sub.loc[d, "coverage_std"] for d in DSORDER]
        axes[0].bar(x + (i - 3) * w, cov, w, yerr=err, capsize=2,
                    label=DISPLAY.get(m,m), color=PALETTE[m], edgecolor="white", linewidth=0.4)
    axes[0].axhline(0.90, color="black", ls="--", lw=1.5, label="target 0.90")
    axes[0].set_xticks(x); axes[0].set_xticklabels([DSLABEL[d] for d in DSORDER])
    axes[0].set_ylabel("Empirical coverage"); axes[0].set_ylim(0, 1.05)
    axes[0].set_title("(a) Marginal coverage (target = 0.90)")

    for i, m in enumerate(ORDER):
        sub = df[df.method == m].set_index("dataset")
        wd = [sub.loc[d, "width_norm_mean"] for d in DSORDER]
        axes[1].bar(x + (i - 3) * w, wd, w, color=PALETTE[m],
                    edgecolor="white", linewidth=0.4, label=DISPLAY.get(m,m))
    axes[1].set_xticks(x); axes[1].set_xticklabels([DSLABEL[d] for d in DSORDER])
    axes[1].set_ylabel("Normalized interval width")
    axes[1].set_title("(b) Interval width (lower = sharper)")
    axes[1].legend(fontsize=8, ncol=2, loc="upper right")
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_coverage_width.png"), bbox_inches="tight")
    plt.close(fig)


def fig3_conditional():
    df = pd.read_csv(os.path.join(RES, "main_results.csv"))
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    x = np.arange(len(DSORDER)); w = 0.115
    for i, m in enumerate(ORDER):
        sub = df[df.method == m].set_index("dataset")
        ws = [sub.loc[d, "worst_slice_mean"] for d in DSORDER]
        ax.bar(x + (i - 3) * w, ws, w, color=PALETTE[m], label=DISPLAY.get(m,m),
               edgecolor="white", linewidth=0.4)
    ax.axhline(0.90, color="black", ls="--", lw=1.5, label="target 0.90")
    ax.set_xticks(x); ax.set_xticklabels([DSLABEL[d] for d in DSORDER])
    ax.set_ylabel("Worst slice (conditional) coverage")
    ax.set_title("Worst slice coverage: reliability across subpopulations")
    ax.legend(fontsize=8, ncol=2, loc="lower right"); ax.set_ylim(0, 1.0)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_conditional_coverage.png"), bbox_inches="tight")
    plt.close(fig)


def fig4_adaptive():
    df = pd.read_csv(os.path.join(RES, "adaptive_curve.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for strat, col in [("adaptive", "#d1495b"), ("random", "#00798c")]:
        s = df[df.strategy == strat].sort_values("budget")
        axes[0].plot(s.budget, s.width, "-o", color=col, label=strat, lw=2)
        axes[0].fill_between(s.budget, s.width - s.width_std, s.width + s.width_std,
                             color=col, alpha=0.15)
        axes[1].plot(s.budget, s.coverage, "-o", color=col, label=strat, lw=2)
    axes[0].set_xlabel("Calibration budget"); axes[0].set_ylabel("Mean interval width")
    axes[0].set_title("(a) Interval width vs budget"); axes[0].legend()
    axes[1].axhline(0.90, color="black", ls="--", lw=1.3, label="target 0.90")
    axes[1].set_xlabel("Calibration budget"); axes[1].set_ylabel("Coverage")
    axes[1].set_title("(b) Coverage vs budget"); axes[1].legend()
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_adaptive_sampling.png"), bbox_inches="tight")
    plt.close(fig)


def fig5_posterior():
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    target = 0.90
    p = np.linspace(0.80, 1.0, 500)
    scenarios = [(45, 50, "#edae49", "small cal. (n=50)"),
                 (180, 200, "#00798c", "medium cal. (n=200)"),
                 (470, 500, "#d1495b", "large cal. (n=500)")]
    prior = 20.0
    for k, n, col, lab in scenarios:
        a = prior * target + k
        b = prior * (1 - target) + (n - k)
        ax.plot(p, stats.beta.pdf(p, a, b), color=col, lw=2.2, label=lab)
        ax.fill_between(p, 0, stats.beta.pdf(p, a, b), color=col, alpha=0.12)
    ax.axvline(target, color="black", ls="--", lw=1.5, label="target 0.90")
    ax.set_xlabel("True coverage probability p")
    ax.set_ylabel("Posterior density")
    ax.set_title("Bayesian Beta Binomial posterior over achieved coverage")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, "fig5_bayesian_posterior.png"), bbox_inches="tight")
    plt.close(fig)


def fig6_intervals():
    # Reproduce ACBUQ vs split-conformal intervals on the degradation task.
    import sys
    sys.path.insert(0, HERE)
    from acbuq.datasets import get_degradation
    from acbuq import methods as M
    from sklearn.model_selection import train_test_split
    X, y, _ = get_degradation()
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.5, random_state=3)
    Xcal, Xte, ycal, yte = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=3)
    lo_a, hi_a, _ = M.acb_uq(Xtr, ytr, Xcal, ycal, Xte, yte, 0.10, seed=3)
    lo_s, hi_s, pred = M.split_conformal(Xtr, ytr, Xcal, ycal, Xte, 0.10, seed=3)
    order = np.argsort(pred)
    idx = order[::2]
    xx = np.arange(len(idx))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    for ax, (lo, hi, name, col) in zip(
            axes, [(lo_s, hi_s, "Split conformal", "#00798c"),
                   (lo_a, hi_a, "ACBUQ (ours)", "#d1495b")]):
        ax.fill_between(xx, lo[idx], hi[idx], color=col, alpha=0.25,
                        label="prediction interval")
        ax.scatter(xx, yte[idx], s=12, color="#2e4057", label="true RUL", zorder=3)
        ax.plot(xx, pred[idx], color=col, lw=1.4, label="point prediction")
        cov = np.mean((yte >= lo) & (yte <= hi))
        ax.set_title(f"{name}  (coverage = {cov:.2f})")
        ax.set_xlabel("Test units (sorted by prediction)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Remaining useful life")
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, "fig6_interval_illustration.png"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig1_architecture(); print("fig1 ok")
    fig2_coverage_width(); print("fig2 ok")
    fig3_conditional(); print("fig3 ok")
    fig4_adaptive(); print("fig4 ok")
    fig5_posterior(); print("fig5 ok")
    fig6_intervals(); print("fig6 ok")
