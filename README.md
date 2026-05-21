# SDA-Logits: Self-Distillation + Deep CORAL for Skin Lesion Classification

PyTorch implementation of a robust skin lesion classification framework using:

- Self-Distillation (SD)
- Deep CORAL
- Feature Distillation
- Multi-exit ResNet18
- Domain Adaptation

The framework is evaluated on the ISIC2018 dataset under domain-shift conditions using turbidity-corrupted images.

---

## Overview

This project proposes a domain adaptation framework for skin lesion classification that improves robustness against image degradation and domain shifts.

The model combines:

1. Classification Supervision
2. Logits-based Self-Distillation
3. Feature-level Distillation
4. Deep CORAL Alignment

using a multi-scale ResNet18 architecture.

---

## Architecture

### Main Components

- ResNet18 backbone
- Multi-exit classifiers (L1–L4)
- Teacher-Student Self-Distillation
- Feature Adapter Module
- Deep CORAL Domain Alignment

---

## Method

### 1. Logits Self-Distillation

Shallow classifiers learn from the deepest classifier outputs.

\[
\mathcal{L}_{SD}=KL(\sigma(z_s/T), \sigma(z_t/T)) \cdot T^2
\]

Where:

- \(z_s\): student logits
- \(z_t\): teacher logits
- \(T\): temperature scaling

---

### 2. Deep CORAL

Feature distributions between source and target domains are aligned.

\[
\mathcal{L}_{CORAL}=\frac{1}{4d^2}\|C_s-C_t\|_F^2
\]

Where:

- \(C_s\): source covariance
- \(C_t\): target covariance

---

### 3. Feature Self-Distillation

Intermediate feature maps are aligned with the deepest feature representation.

\[
\mathcal{L}_{feat}=\sum_i \|f_i-f_t\|_2^2
\]

---

## Dataset

### Source Domain

- ISIC2018

### Target Domain

- Turbidity-corrupted ISIC2018
- Example:
  - `isic2018_turbidity_medium_center`

---

## Project Structure

```bash
.
├── main.py
├── model_SDA.py
├── coral.py
├── data_loader.py
├── utils.py
├── classification_report_*.csv
├── sda_logits.csv
└── tsne_*.png
```

---

## Installation

### Requirements

```bash
pip install torch torchvision timm scikit-learn pandas matplotlib tqdm
```

---

## Training

### Source Only

```bash
python main.py \
    --source isic2018 \
    --target isic2018_turbidity_medium_center
```

### Domain Adaptation

```bash
python main.py \
    --source isic2018 \
    --target isic2018_turbidity_medium_center \
    --lambda_coral 0.1 \
    --lambda_sd_logits 0.4 \
    --lambda_sd_feat 0.1
```

---

## Important Arguments

| Argument | Description | Default |
|---|---|---|
| `--epochs` | Number of epochs | 10 |
| `--batch_size` | Batch size | 8 |
| `--lr` | Learning rate | 1e-3 |
| `--lambda_coral` | CORAL loss weight | 0.1 |
| `--lambda_sd_logits` | Logits SD weight | 0.4 |
| `--lambda_sd_feat` | Feature SD weight | 0.1 |
| `--temperature` | Distillation temperature | 1.0 |

---

## Evaluation Metrics

The framework reports:

- classification report (ACC, PRE, REC, F1)
- tsne

Classification reports are automatically saved as CSV files.

---

## t-SNE Visualization

The project provides feature visualization using t-SNE.

Generated outputs include:

- Source domain features
- Target domain features
- Adaptation quality comparison

---

## Experimental Pipeline

```text
Source Images ──┐
                ├── ResNet18 + Multi-Exit Heads
Target Images ──┘
                        │
                        ├── Classification Loss
                        ├── Logits Self-Distillation
                        ├── Feature Distillation
                        └── Deep CORAL Alignment
```

---

## Output Files

```bash
classification_report_adaptation_logits.csv
sda_logits.csv
tsne_adaptation_run0_SDA_logits.png
```

---

## Acknowledgements

- ISIC 2018 Challenge Dataset
- PyTorch
- Deep CORAL
  
### License
**MIT license**
