# Traffic Sign Classifier (GTSRB) — ADAS Perception Project

A CNN-based traffic sign classifier, built from the ground up to learn computer vision and deep learning fundamentals for automotive perception / ADAS applications. This project complements [Racing Line Optimizer](https://github.com/yessin-hamdi/racing-line-optimizer) (mathematics, optimization, algorithms) by covering the AI / computer vision side of an automotive-AI portfolio.

## Status:  V1 complete

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

Full analysis and charts: [`notebooks/02_training_results.ipynb`](notebooks/02_training_results.ipynb), [`notebooks/03_v2_results.ipynb`](notebooks/03_v2_results.ipynb), [`notebooks/04_v3_results.ipynb`](notebooks/04_v3_results.ipynb), [`notebooks/05_evaluation.ipynb`](notebooks/05_evaluation.ipynb)

## Limitations

- The imbalance between training images per class (150 to 1,500) leaves a gap between the macro F1 (0.898) and weighted F1 (0.919), even after regularization.
- The errors this model encounters are far from random — they split into two explainable visual clusters: speed limit digit signs (confusion between the "2" and "3" in 20km/h and 30km/h signs), and triangular signs sharing the same shape/color but differing only in small pictograms. This points to a resolution limitation of our 32×32 input.
- This model was only trained and tested on GTSRB images — real-world robustness (occlusion, night driving, adverse weather, non-German signage) is untested.
- The model is fairly small, with only 273K parameters, due to the hardware constraints it was trained on.
- Results come from a single train/test split, not cross-validation — 91.91% is one measurement, not a statistically averaged estimate.

## Future Improvements

- Directly target the two error clusters identified above, via class-weighted loss or oversampling the rare classes.
- Potentially use a higher-resolution input to help resolve fine details (digits, pictograms).
- Improve the architecture with batch normalization and a deeper network.
- Extend toward real ADAS perception: moving from classification (pre-cropped signs) to detection (locating signs within a full road scene).