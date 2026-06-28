# ACBUQ: Adaptive Conformal Bayesian Uncertainty Quantification

Reproducible code and data for the manuscript *Adaptive Conformal Bayesian Uncertainty
Quantification for Reliable High Dimensional Intelligent Systems*, submitted to the
*Annals of Operations Research* special issue on Advances in Reliability and Statistical
Computing for Intelligent Systems.

ACBUQ equips any black box predictor with prediction intervals that carry a finite sample,
distribution free coverage guarantee, makes the interval width adapt to local difficulty
through conformalized quantile regression, and certifies the achieved coverage with a
Bayesian Beta Binomial layer that also sets a small, data driven safety margin.

## Method in one paragraph

A neural network surrogate estimates the system mean and a gradient boosted pair estimates
the conditional quantiles. Conformity scores on a held out calibration split are used to
correct the quantile interval so that marginal coverage holds regardless of model
misspecification. The number of covered calibration points is treated as a Binomial outcome,
yielding a Beta posterior over the true coverage probability; from this posterior we report a
credible interval, the probability of meeting the target, and a certified effective level
that shrinks the safety margin as calibration data grow.

## Installation

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.11 and the versions pinned in `requirements.txt`.

## Reproducing the results

```bash
# 1. Run the full benchmark (10 seeds x 3 datasets x 7 methods) and the adaptive study.
python experiments/run_benchmark.py

# 2. Regenerate all six figures from the results CSVs.
python experiments/make_figures.py
```

Outputs are written to `results/` (CSV) and `figures/` (PNG). Running the benchmark
overwrites the bundled CSVs with identical numbers because every dataset and split is seeded.

## Repository layout

```
acbuq/
  datasets.py     real Diabetes loader plus seeded Friedman1 and bearing degradation simulators
  methods.py      surrogate, split conformal, CQR, Bayesian Beta Binomial certification,
                  adaptive sampling, and the six baselines; coverage and width metrics
experiments/
  run_benchmark.py   main benchmark and adaptive sampling study
  make_figures.py    builds Figures 1 to 6
results/          benchmark outputs (CSV) used by the manuscript tables
figures/          colourful figures embedded in the manuscript
```

## Datasets

| Dataset      | Domain                       | Type      | Dim | Samples |
|--------------|------------------------------|-----------|-----|---------|
| Diabetes     | Medical decision support     | Real      | 10  | 442     |
| Friedman one | High dimensional surrogate   | Simulated | 10  | 2000    |
| Degradation  | Industrial condition monitor | Simulated | 12  | 400     |

The Diabetes data ships with scikit learn (Efron et al., 2004). The two simulators are fully
seeded and regenerate exactly.

## Headline results (90 percent target, 10 seeds)

ACBUQ meets or exceeds the target on every task and achieves the best worst slice
(conditional) coverage of all seven methods, while naive deep ensembles under cover badly.
See `results/main_results.csv` for the complete table.

## License

Released under the MIT License. See `LICENSE`.
