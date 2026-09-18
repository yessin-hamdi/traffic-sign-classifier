# Traffic Sign Classifier (GTSRB) — ADAS Perception Project

A CNN-based traffic sign classifier, built from the ground up to learn computer vision and deep learning fundamentals for automotive perception / ADAS applications. This project complements [Racing Line Optimizer](https://github.com/yessin-hamdi/racing-line-optimizer) (mathematics, optimization, algorithms) by covering the AI / computer vision side of an automotive-AI portfolio.

## Status:  In Progress

Regularized CNN trained with data augmentation, dropout, and early stopping — 91.91% test accuracy. Currently moving into full evaluation (confusion matrix, per-class metrics).

## Dataset

[German Traffic Sign Recognition Benchmark (GTSRB)](https://benchmark.ini.rub.de/) — 43 traffic sign classes, ~26,600 training images / 12,630 test images, real photographs (varying lighting, angle, blur). Images resized to 32×32 RGB. Note: classes are imbalanced (150–1,500 images per class).

## Setup

```bash
git clone https://github.com/yessin-hamdi/trafic-sign-project.git
cd trafic-sign-project
python -m venv venv
source venv/Scripts/activate    # Windows Git Bash
pip install -r requirements.txt
```

## Usage

```bash
python src/dataset.py     # downloads GTSRB, verifies loading
python src/visualize.py   # sample images + class distribution
python src/train.py       # trains the CNN with augmentation/dropout/early stopping, saves best checkpoint
```

## Architecture

Simple CNN: `Input(32×32×3) → Conv(3→16)+ReLU+Pool → Conv(16→32)+ReLU+Pool → Flatten → FC(128)+Dropout(0.5) → FC(43)`. ~273K parameters.

**Training setup:** data augmentation (random rotation ±10°, translation ±10%, color jitter) applied to training data only; dropout (p=0.5) on the FC layer; early stopping (patience=5) saving only the best-performing checkpoint.

## Results

| Version | Peak test accuracy | Notes |
|---|---|---|
| v1 — baseline | 85.64% | No regularization; clear overfitting (train loss to 0.04 while accuracy plateaued/degraded) |
| v2 — + augmentation, dropout, early stopping | 89.63% | Regularized, but epoch budget (30) capped before convergence |
| v3 — same as v2, full epoch budget | 91.91% | Early stopping triggered naturally at epoch 49 (best: epoch 44) |

Full analysis and charts: [`notebooks/02_training_results.ipynb`](notebooks/02_training_results.ipynb), [`notebooks/03_v2_results.ipynb`](notebooks/03_v2_results.ipynb), [`notebooks/04_v3_results.ipynb`](notebooks/04_v3_results.ipynb).

## Roadmap

- [x] Early stopping (save best model, not just final epoch)
- [x] Data augmentation
- [x] Dropout regularization
- [ ] Full evaluation: confusion matrix, per-class precision/recall
- [ ] Final write-up and limitations
