# Traffic Sign Classifier (GTSRB) — ADAS Perception Project

A CNN-based traffic sign classifier, built from the ground up to learn computer vision and deep learning fundamentals for automotive perception / ADAS applications. This project complements [Racing Line Optimizer](https://github.com/yessin-hamdi/racing-line-optimizer) (mathematics, optimization, algorithms) by covering the AI / computer vision side of an automotive-AI portfolio.

## Status: 🚧 In Progress

Baseline CNN trained and evaluated. Currently working on regularization (dropout, data augmentation, early stopping) to address overfitting observed in the baseline.

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
python src/train.py       # trains the CNN, saves weights
```

## Architecture

Simple CNN: `Input(32×32×3) → Conv(3→16)+ReLU+Pool → Conv(16→32)+ReLU+Pool → Flatten → FC(128) → FC(43)`. ~273K parameters.

## Results so far (Baseline v1)

- Test accuracy: 84.88% (peak: 85.64% at epoch 6)
- Clear overfitting past epoch 6 (training loss keeps dropping, test accuracy plateaus/degrades)
- Full analysis: [`notebooks/02_training_results.ipynb`](notebooks/02_training_results.ipynb)

## Roadmap

- [ ] Early stopping (save best model, not just final epoch)
- [ ] Data augmentation
- [ ] Dropout regularization
- [ ] Full evaluation: confusion matrix, per-class precision/recall
- [ ] Final write-up and limitations
