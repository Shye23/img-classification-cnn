# Breast Cancer Histopathology Image Classification

## Project Overview

This project implements a deep learning pipeline for breast cancer histopathology image classification using transfer learning in PyTorch. The codebase contains preprocessing, dataset preparation, model definition, training, and evaluation modules designed for reproducible experiments and clear metric logging.

Key features:
- Patient-wise train/validation split to prevent data leakage
- Transfer learning with configurable architectures (e.g., ResNet-18, MobileNetV2)
- Per-epoch logging and persistent metric artifacts (CSV, JSON, confusion matrices)
- Modular `src/` structure for easy experimentation

---

## Dataset

This project uses the Breast Cancer Histopathology Images dataset (IDC dataset) available on Kaggle.

Dataset provided by: Paul Mooney
Source: https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images

The dataset contains labeled histopathology image patches categorized into cancerous (IDC-positive) and non-cancerous classes.

---

## Repository Structure

```
project/
├── data/
├── notebooks/
├── src/
│   ├── preprocess.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
   └── evaluate.py
├── outputs/
└── requirements.txt
```

Files of interest:
- `src/train.py` — training loop, saves models and writes per-epoch metrics
- `src/evaluate.py` — evaluation routine that computes classification report and confusion matrix and saves JSON/TXT artifacts
- `src/utils.py` — helpers for logging, CSV append, and JSON saving
- `outputs/` — default output directory where models and metric artifacts are stored

---

## Quick Setup

Prerequisites:
- Python 3.10 (project tested with 3.10)
- A virtual environment (one exists in `torch_env/` in this workspace)
- CUDA-enabled PyTorch if you plan to train on GPU

Install dependencies:

```bash
# from project root
python -m pip install -r requirements.txt
```

Activate your virtual environment (Windows PowerShell example):

```powershell
(Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned)
& .\torch_env\Scripts\Activate.ps1
```

---

## How to Run Training

Start training from the project root:

```bash
python src/train.py
```

What the script does:
- Loads train/validation loaders from `data/dataset_final/train` and `data/dataset_final/val`.
- Runs a training loop for the configured number of epochs.
- Computes validation metrics each epoch via `src/evaluate.py`.
- Persists per-epoch metrics and artifacts to `outputs/`.

---

## Logged Artifacts & Where to Find Them

Per-epoch and final artifacts are written to `outputs/`:

- `outputs/training.log` — chronological training and evaluation log (also printed to console).
- `outputs/metrics.csv` — appended row per epoch with columns: `epoch, train_loss, train_accuracy, val_loss, val_accuracy`.
- `outputs/model.pth` — latest model checkpoint (overwritten each epoch).
- `outputs/best_model.pth` — best-performing model on validation (saved when validation accuracy improves).
- `outputs/training_summary.json` — final run summary containing best epoch and paths.
- `outputs/validation_metrics_epoch_XXX.json` — per-epoch validation metrics (loss, accuracy, classification report text, confusion matrix numeric data).
- `outputs/confusion_matrix_epoch_XXX.json` — per-epoch confusion matrix with labels.
- `outputs/classification_report_epoch_XXX.txt` — human-readable sklearn classification report per epoch.

These artifacts let you track improvement across epochs and reproduce results later.

---

## Evaluation & Interpretation

Per-epoch validation metrics include:
- Validation loss and accuracy (scalar)
- Full classification report (precision, recall, f1-score per class)
- Confusion matrix (saved as JSON and as TXT report)

Recommended monitoring:
- Use `outputs/metrics.csv` to plot trends for loss and accuracy across epochs.
- Inspect `classification_report` and the confusion matrix to check per-class performance (particularly recall for the cancer class).

Example quick plot (Python snippet):

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('outputs/metrics.csv')
plt.plot(df['epoch'], df['train_loss'], label='train_loss')
plt.plot(df['epoch'], df['val_loss'], label='val_loss')
plt.legend()
plt.show()
```

---

## Reproducibility

- Train/validation splits are created with a fixed random seed in preprocessing to ensure reproducible experiments.
- `training_summary.json` records the best epoch and file locations for reproducibility.
- Save hyperparameters and any notable environment details alongside run artifacts if you intend to compare many runs.

---

## Results Summary (from experiments)

The project includes the following summarized outcome from earlier subset experiments:

- Achieved ~92% validation accuracy on a subset with high recall (~0.95) for the cancer class.
- Example confusion matrix (subset):

```
[[ 436  107]
 [ 102 1973]]
```

Notes: accuracy alone is misleading for medical tasks—focus on recall/precision/F1 per-class, especially recall for cancer detection.

---

## Next Steps & Tips

- If you want experiment tracking across runs, add a simple experiment ID and save a copy of `metrics.csv` per-run (e.g., `metrics_run_<id>.csv`).
- Consider integrating TensorBoard or Weights & Biases for interactive dashboards:

```bash
pip install tensorboard
# then add SummaryWriter hooks in src/train.py and src/evaluate.py
```

- For long training runs, checkpoint periodically with epoch number included (e.g., `model_epoch_{:03d}.pth`).

---

## Contact / Attribution

If you want help extending the README with reproducible experiment recipes or a small tutorial notebook that demonstrates plotting and interpreting metrics, open an issue or request and I can add it.

---

Appendix: Short run checklist

1. Activate virtualenv
2. Install `requirements.txt`
3. Ensure `data/dataset_final/train` and `data/dataset_final/val` exist and are populated
4. Run `python src/train.py`
5. Inspect `outputs/metrics.csv`, `outputs/training.log`, and `outputs/classification_report_epoch_*.txt`
