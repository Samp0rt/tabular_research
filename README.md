# tabular_research

Implementation of the paper "Tabular Synthesis Reinvention: LLM Capabilities for Creating Realistic Multidimensional Distributions"

## Overview

This repository provides tools for generating synthetic tabular data, automating experiments, evaluating synthetic data quality, and optimizing hyperparameters. Both classical and modern deep learning/LLM approaches are supported.

## Project Structure

- `scripts/fit_and_generate.py` — Main script for training generative models and producing synthetic datasets using various plugins.
- `scripts/generator_syn.py` — Synthetic data generation using LLMs based on real data samples.
- `scripts/synthcity_metrics.py` — Automatic calculation of synthetic data quality metrics for generated datasets.
- `scripts/modify_detection.py` — Modifies detection metric results for correct comparison.
- `scripts/vectgan.py` — Implementation of the VECTGAN plugin for the synthcity library.
- `data/` — Original datasets split into train/test and a `data_info.csv` file with metadata.
- `optuna/` — Hyperparameter optimization results for different models and datasets.
- `docker/requirements.txt` — Project dependencies.
- `docker/Dockerfile` — Docker image for reproducible experiments.
- `generated/` — Output directory for generated synthetic data.

## Quick Start

### 1. Clone the repository

```bash
git clone <repository_url>
cd tabular_research
```

### 2. Run with Docker (recommended)

```bash
cd docker
docker build -t tabular_research .
docker run --gpus all -it -v $(pwd)/../:/synth_tests tabular_research
```

### 3. Manual installation

```bash
pip install -r docker/requirements.txt
```

### 4. Running experiments

```bash
python scripts/fit_and_generate.py --data_folder ./data --output_folder ./results --optuna_params ./optuna --repeats 5
```

Arguments:
- `--data_folder` — Path to the data directory (default: `./data`)
- `--output_folder` — Directory to save results (default: `./results`)
- `--optuna_params` — Path to directory with optimized hyperparameters
- `--repeats` — Number of synthetic data generations per dataset/model

### 5. LLM-based synthetic data generation

```bash
python scripts/generator_syn.py
```
This script allows you to integrate your own LLM client for generating synthetic data based on real samples.

### 6. Synthetic data quality evaluation

```bash
python scripts/synthcity_metrics.py --data_folder ./data --output_folder ./generated --repeats 5
```
This script automatically calculates quality metrics for all generated datasets.

## Example Data Structure

The `data/data_info.csv` file contains metadata for each dataset:

| df_name | target_name | task_type      | numeric_cols_indxs | row_number |
|---------|-------------|---------------|--------------------|------------|
| ...     | ...         | ...           | ...                | ...        |

## Supported Models

- CTGAN, TVAE, DDPM, DPGAN, ADSGAN, Bayesian Network, VECTGAN, and others
- LLM-based generation (plug in your own client)

## Dependencies

See `docker/requirements.txt` for the full list. Key dependencies include:
- Python 3.11
- torch, pandas, scikit-learn, synthcity, transformers, and more

## Containerization

The provided Dockerfile is based on a CUDA-enabled Ubuntu image for GPU-accelerated training and generation.

## License

This project is intended for research purposes only.

---

For questions or contributions, please open an issue or submit a pull request. 