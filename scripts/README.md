# FedCLCM Chapter-4 Scripts

## Purpose

This directory reorganizes the thesis experiments around the structure of a Chinese master's thesis Chapter 4.

Every experiment script writes outputs to `thesis_log/<experiment_name>/` and generates:

- `summary_<timestamp>.csv`
- `summary_<timestamp>.json`
- `curves_<timestamp>/`

The `curves` directory stores per-round CSV files and is intended to be used directly for thesis plots such as convergence curves, ASR curves, and trade-off figures.

## Script Mapping

### `00_prepare_datasets.sh`

Prepare trigger assets and regenerate the datasets used in Chapter 4.

It uses the existing `dataset/utils/generate*.py` entry points through a compatibility wrapper so that old hardcoded assumptions do not break the new experiment layout.

Default behavior:

- keeps existing datasets if the config already matches
- set `FORCE_REBUILD=1` to rebuild from scratch

### `01_main_results_basic.sh`

Main comparison experiments for the basic benchmark setting:

- MNIST
- FashionMNIST
- CIFAR-10
- static BadNet attack
- comparison baselines aligned with the current codebase implementation (`FedAvg`, `FedRep`, `FedMedian`, `FedTrimmed`, `FedBulyan`, `FedFLIP`, `FedCLCM`)

Suggested thesis subsection:

- 4.2 Main Results on Standard Benchmarks

### `01_main_results_pfedba.sh`

Compatibility wrapper that forwards to `01_main_results_basic.sh`.

### `02_attack_generalization_static.sh`

Attack-generalization experiments on additional static attacks after the main BadNet benchmark:

- Blend
- SIG

Suggested thesis subsection:

- 4.3 Generalization Across Backdoor Attack Types

### `03_module_ablation.sh`

Core module ablation experiments, including:

- CIFAR-10 8-group ablation
- FashionMNIST 4-group cross-dataset ablation

Suggested thesis subsection:

- 4.4 Ablation Study

Note:

- this script still uses PFedBA as an advanced backdoor setting for module validation rather than as the chapter's lead comparison benchmark

### `04_client_sensitivity.sh`

Sensitivity study for client-side modules:

- `lambda_cl`
- `aug_strength`

Suggested thesis subsection:

- 4.5 Sensitivity of Client-Side Components

### `05_server_sensitivity.sh`

Sensitivity study for server-side modules:

- `mask_tau`
- `mask_alpha`
- `rt_beta`

Suggested thesis subsection:

- 4.6 Sensitivity of Server-Side Components

### `06_scalability_robustness.sh`

Scalability and robustness study for:

- client count
- malicious client count
- join ratio

Suggested thesis subsection:

- 4.7 Scalability and Robustness

### `07_heterogeneity_study.sh`

Heterogeneity study across IID and several Dirichlet settings.

Suggested thesis subsection:

- 4.8 Data Heterogeneity Analysis

### `08_efficiency_overhead.sh`

Efficiency comparison under a shorter static CIFAR-10 BadNet run.

Suggested thesis subsection:

- 4.9 Efficiency and Overhead

### `09_run_all_chapter4.sh`

One-click runner for the full Chapter-4 experiment suite.

## Usage

### Prepare datasets

```bash
bash scripts/00_prepare_datasets.sh
```

### Rebuild datasets from scratch

```bash
FORCE_REBUILD=1 bash scripts/00_prepare_datasets.sh
```

### Run a single section

```bash
GPUS="0 1 2 3" bash scripts/03_module_ablation.sh
```

### Run the full Chapter 4 suite

```bash
GPUS="0 1 2 3" bash scripts/09_run_all_chapter4.sh
```

## Notes

- The scripts assume the project root is the current repository.
- GPU scheduling is driven by the `GPUS` environment variable.
- The data-generation wrapper intentionally routes all dataset creation through the legacy `generate*.py` scripts so the final datasets remain consistent with the existing codebase conventions.
- `FedRT` has been removed from the comparison scripts because it overlaps with the AdvPurge-style server-side route and should not be treated as an independent baseline here.
