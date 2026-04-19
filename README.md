# Auto-Researcher: Bayesian Hyperparameter Optimization for Transformers

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![Optuna](https://img.shields.io/badge/Optuna-Bayesian_Optimization-00ADFF)](https://optuna.org/)
[![Weights & Biases](https://img.shields.io/badge/W&B-Optimization-FFBE00?logo=weightsandbiases&logoColor=white)](https://wandb.ai/)

**Auto-Researcher** is an intelligent pipeline designed to automate the discovery of optimal transformer architectures. By leveraging **Bayesian Optimization (via Optuna)**, it programmatically explores the hyperparameter space of a character-level transformer to minimize validation loss (Bits-Per-Character).

---

## Overview

Researching deep learning models often involves tedious manual hyperparameter tuning. This project transforms that manual process into an automated, feedback-driven loop:
1. **Researcher** proposes a model configuration (layers, dimensions, dropout).
2. **Trainer** instantiates the model and trains it on text data.
3. **WandB** logs the real-time performance metrics.
4. **Researcher** analyzes the results and refines the search space using Bayesian heuristics.

## Technical Stack

-   **Deep Learning Framework:** [PyTorch](https://pytorch.org/) (Custom Transformer Architecture)
-   **Optimization Engine:** [Optuna](https://optuna.org/) (TPE Sampler for Bayesian Search)
-   **Experiment Tracking:** [Weights & Biases](https://wandb.ai/) (Offline logging mode)
-   **Metrics:** Bits-Per-Character (BPB), Validation Cross-Entropy Loss.

## Architecture

### 1. The Transformer (`train.py`)
A robust character-level Transformer implementation featuring:
-   Multi-head self-attention with scaled dot-product optimization.
-   Learned positional embeddings.
-   Layer normalization and GELU activation functions.
-   Early stopping mechanism to prevent wasted compute.

### 2. The Optimizer (`researcher.py`)
An orchestrator script that:
-   Programmatically edits `train.py` configurations.
-   Executes training as a subprocess.
-   Parses stdout for metrics.
-   Saves comprehensive trial logs to `bayesian_logs.json`.

---

## Project Structure

```text
.
├── Auto_Researcher/
│   ├── researcher.py      # Bayesian optimization orchestrator
│   ├── train.py           # Core Transformer training script
│   ├── data.txt           # Training dataset (TinyShakespeare snippet)
│   ├── bayesian_logs.json # Historical results of optimization trials
│   └── wandb/             # Local experiment logs
└── README.md              # Project documentation
```

## Setup & Installation

Ensure you have Python 3.8+ installed.

```bash
# Clone the repository
git clone https://github.com/your-repo/Auto_Researcher.git
cd Auto_Researcher

# Install dependencies
pip install torch optuna wandb
```

## Usage

### Start Automated Research
To begin the Bayesian search for optimal hyperparameters:
```bash
python researcher.py
```
This will run multiple trials, sequentially updating the model configuration and recording performance.

### Run a Single Training Session
To run the training script with current configuration:
```bash
python train.py
```

---

## Performance Metrics

The project focuses on **Bits-Per-Character (BPB)** as the primary metric for language modeling quality.
-   **Calculation:** `avg_loss / ln(2)`
-   **Target:** Lower is better (higher compression/prediction accuracy).

## License
This project is for educational and internship research purposes.