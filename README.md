# DP_Research_Paper

# K-LSG-Shield

Do layered synthetic-data and self-distillation defences add anything beyond DP-SGD against membership inference attacks?

Karen K. Shaylee, J. Premika, Jai Vinita L
School of Computer Science and Engineering, Vellore Institute of Technology, Chennai, India

## Overview

K-LSG-Shield is a four-layer defence against membership inference attacks (MIAs), built and evaluated as a test case for a single question: when synthetic-data generation, self-distillation, differential privacy, and output sanitization are stacked into one pipeline, does the stack lower worst-case attack leakage beyond DP-SGD alone?

The four layers:

- L1: cluster-based noisy synthetic data, built from k-equivalence clusters of the private data
- L2: cluster-disjoint self-distillation using noisy out-of-bag teacher votes
- L3: a DP-SGD student trained on the synthetic set at a target epsilon of 2
- L4: a rank-preserving output sanitizer applied at query time

The pipeline is evaluated on Adult Income (tabular) and CIFAR-10 (image) against ten attacks (seven static, three defence-aware adaptive attacks), five re-implemented published baselines, DP-SGD at the same student noise budget, and an early-stopped undefended reference model. Results use worst-case attack AUC, balanced attack accuracy, and true-positive rate at low false-positive rates, with paired significance tests and bootstrap confidence intervals over five seeds.

## Repository contents

- `code.ipynb`: the full experimental pipeline. Builds the four defence layers, the five baselines, the ten attacks, and the evaluation protocol; trains all models; and writes the result tables and figures.
- `code_logs.log`: console log from the full run (both datasets, five seeds).
- `results_tables.zip`: per-seed and aggregated result tables (attack AUC, accuracy, TPR at low FPR, ablation tables, computational cost).
- `results_figures.zip`: generated figures (privacy-utility curves, k sweep, epsilon sweep, ablation plots, cost comparison).
- `LICENSE`: MIT license.

## Environment

All experiments ran in a single Kaggle notebook cell on an NVIDIA Tesla T4 GPU.

- Python 3.13.15
- PyTorch 2.11.0 (CUDA 12.8)
- Opacus 1.6.0, default PRV accountant
- scikit-learn 1.6.1
- NumPy 2.1.3
- SciPy 1.16.3
- pandas 2.3.3
- matplotlib 3.10.0

Exact library versions are also recorded automatically with every run inside `code_logs.log`.

## Datasets

- Adult Income (UCI Machine Learning Repository), cleaned to 45,222 records.
- CIFAR-10 (Krizhevsky, 2009), 30,000 images sampled from the official training and test sets.

Both datasets are split into six disjoint, class-stratified partitions: target training, target holdout, defender reference, attacker population, shadow pool, and true test. No defence or attack is evaluated on data outside its assigned partition.

## Running

1. Open `code.ipynb` in Kaggle or a GPU-enabled Jupyter environment.
2. Run all cells in order. The notebook installs its own dependencies, downloads both datasets, trains the undefended reference, all five baselines, DP-SGD, and the K-LSG-Shield sweep over k, then runs all ten attacks against every setting.
3. Result tables are written to the paths referenced in `results_tables.zip`; figures are written to the paths referenced in `results_figures.zip`.

A full run across both datasets and five seeds takes approximately 25,200 seconds on a single T4 GPU.

## Headline result

K-LSG-Shield lowers worst-case static attack AUC from 0.591 to 0.504 on Adult Income and from 0.807 to 0.513 on CIFAR-10, matching the random-guess floor on both datasets. It is tied with DP-SGD at the same student noise budget, which reaches the same floor and has higher accuracy (0.845 against 0.833 on Adult Income, 0.358 against 0.302 on CIFAR-10). The ablation shows that when DP-SGD is present, the DP layer accounts for the privacy gain and the additional layers add no measurable reduction in worst-case leakage.

## Status

This repository accompanies a manuscript under revision. See the paper for the full threat model, methodology, evaluation protocol, limitations, and a list of concrete next steps arising from peer review.

## License

MIT. See `LICENSE`.
