# SDA: Self-Distillation Framework for Domain Adaptation in Skin Lesion Classification

This repository provides the official PyTorch implementation of the proposed **Self-Distillation Framework for Domain Adaptation in Skin Lesion Classification**.

The repository includes the network architecture, dataset loader, domain adaptation modules, evaluation pipeline, and utility functions used in our experiments. The main executable script is `test_SDA_logits.py`, which performs the complete evaluation procedure using a trained SDA model.

---

# Overview

Deep learning models for skin lesion classification often suffer performance degradation when applied to images collected under different imaging conditions or from different domains. To improve robustness, we propose a self-distillation framework for domain adaptation.

The proposed framework employs a ResNet-18 backbone with auxiliary classifiers attached to intermediate layers. During training, the deepest classifier transfers knowledge to the intermediate classifiers through self-distillation. Domain discrepancy between the source and target datasets is further reduced through feature alignment.

This repository contains the implementation used in our study for model evaluation and reproducibility.

---

# Repository Structure

```text
SDA/
├── LICENSE
├── README.md
├── coral.py              # Feature alignment loss
├── data_loader.py        # Dataset loading and preprocessing
├── model_SDA.py          # SDA network architecture
├── test_SDA_logits.py    # Main evaluation script
└── utils.py              # Utility functions
```

---

# Dataset Information

The experiments were conducted using the following publicly available skin lesion datasets.

---
**ISIC 2018: Skin Lesion Analysis Towards Melanoma Detection**

Official website

https://challenge.isic-archive.com/data/

Reference

Codella NCF, et al.
*Skin Lesion Analysis Toward Melanoma Detection: A Challenge at the 2018 ISIC Workshop.*

---

**SKINL2 Dataset**

Official website

https://www.it.pt/AutomaticPage?id=3459

Please download both datasets from their official websites before running the code.

---

# Data Preprocessing

Image preprocessing is implemented in `data_loader.py`.

The preprocessing pipeline includes:

- Image resizing
- Image normalization
- Conversion to PyTorch tensors
- Dataset loading for source and target domains

Please modify the dataset paths according to your local environment before running the code.

---

# Methodology

The proposed SDA framework consists of the following components.

- ResNet-18 backbone
- Multi-level feature extraction
- Auxiliary classifiers
- Self-distillation between intermediate and final classifiers
- Feature alignment module for domain adaptation
- Final classification layer

During training, the deepest classifier serves as the teacher and transfers knowledge to shallower classifiers through logit-based self-distillation. Feature alignment is employed to reduce the discrepancy between the source and target domains.

---

# Code Information

## test_SDA_logits.py

This is the main executable script of the repository.

The script performs the following tasks:

- Loads the source and target datasets
- Constructs the SDA model
- Loads a trained model checkpoint
- Performs model evaluation
- Reports classification performance

Running this script reproduces the evaluation pipeline described in the manuscript.

---

## model_SDA.py

Defines the SDA network architecture, including:

- ResNet-18 backbone
- Intermediate feature extraction
- Auxiliary classifiers
- Feature projection modules
- Self-distillation modules

---

## data_loader.py

Loads the ISIC 2018 and SKINL2 datasets and performs image preprocessing.

---

## coral.py

Implements the feature alignment loss used for domain adaptation.

---

## utils.py

Provides utility functions for evaluation and experiment management.

---

# Requirements

The implementation was developed using **Python 3.10** and **PyTorch**.

All required Python packages are listed in the `requirements.txt` file.

Install the dependencies using:

```bash
pip install -r requirements.txt
```

---

# Usage

## Step 1. Download the datasets

Download the ISIC 2018 and SKINL2 datasets from their official websites.

---

## Step 2. Configure the dataset paths

Modify the dataset paths and model checkpoint path in `test_SDA_logits.py` according to your local environment.

---

## Step 3. Run the evaluation

```bash
python test_SDA_logits.py
```

The script automatically:

- loads the datasets,
- initializes the SDA model,
- loads the trained model checkpoint,
- performs evaluation,
- reports the classification results.

---

# Reproducibility

To reproduce the experimental results reported in the manuscript:

1. Download the ISIC 2018 and SKINL2 datasets.
2. Configure the dataset paths.
3. Place the trained model checkpoint in the specified directory.
4. Execute

```bash
python test_SDA_logits.py
```

The implementation follows the evaluation procedure described in the manuscript.

---

# Citation

If you use this repository in your research, please cite:

```bibtex
@article{YOUR_PAPER,
  title={Self-Distillation Framework for Domain Adaptation in Skin Lesion Classification},
  author={Author(s)},
  journal={Journal Name},
  year={2026}
}
```

Please also cite the original ISIC 2018 and SKINL2 dataset publications.

---

# License

This project is distributed under the MIT License.

Please refer to the `LICENSE` file for details.

---

# Acknowledgements

We gratefully acknowledge the creators of the following publicly available datasets:

- ISIC 2018 Challenge Dataset
- SKINL2 Dataset

Their contributions have greatly supported research on automated skin lesion analysis.
