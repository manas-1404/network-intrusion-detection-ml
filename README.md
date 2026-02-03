# Network Intrusion Detection using Machine Learning

A machine learning system for detecting network intrusions using the NSL-KDD dataset. The model classifies network traffic as either normal or one of multiple attack types.

## Overview

This project implements a network intrusion detection system (NIDS) using Random Forest. It's designed to identify malicious network activity by analyzing connection-level features extracted from network traffic.

## Dataset

The project uses the **NSL-KDD dataset**, an improved version of the original KDD Cup 1999 dataset. It addresses some of the inherent problems of the original dataset including redundant records.

- **Training samples:** 125,973
- **Test samples:** 22,544
- **Features:** 41 (38 numerical + 3 categorical)
- **Attack types in training:** 23 classes (22 attacks + normal)

### Attack Categories

| Category | Attack Types |
|----------|-------------|
| DoS | back, land, neptune, pod, smurf, teardrop |
| Probe | ipsweep, nmap, portsweep, satan |
| R2L | ftp_write, guess_passwd, imap, multihop, phf, spy, warezclient, warezmaster |
| U2R | buffer_overflow, loadmodule, perl, rootkit |

**Note:** The test set contains 15 novel attack types not present in training data (apache2, mailbomb, mscan, etc.).

## Installation

```bash
# Clone the repository
git clone https://github.com/manas-1404/network-intrusion-detection.git
cd network-intrusion-detection

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Install dependencies
pip install pandas numpy scikit-learn xgboost optuna kagglehub matplotlib seaborn joblib
```

## Usage

### Training the Model

```bash
python train_random_forest_classifier.py
```

This will:
1. Download the NSL-KDD dataset via kagglehub
2. Preprocess and encode features
3. Train a Random Forest classifier
4. Evaluate on both all test samples and known attacks only
5. Save the model, encoders, and performance reports

### Using the Trained Model

```python
import joblib

# Load model and encoders
model = joblib.load('random_forest_classifier_model/rf_model.pkl')
feature_encoder = joblib.load('random_forest_classifier_model/feature_encoder.pkl')
label_encoder = joblib.load('random_forest_classifier_model/label_encoder.pkl')

# Preprocess new data
X_encoded = feature_encoder.transform(X_new)

# Predict
predictions = model.predict(X_encoded)
labels = label_encoder.inverse_transform(predictions)
```

## Project Structure

```
network-intrusion-detection/
├── data_loader.py                 # Dataset loading and preprocessing
├── encoder.py                     # Feature and label encoding wrappers
├── model.py                       # Model wrappers (RandomForest, XGBoost)
├── train_random_forest_classifier.py  # Training pipeline
├── random_forest_classifier_model/    # Saved model artifacts
│   ├── rf_model.pkl
│   ├── feature_encoder.pkl
│   ├── label_encoder.pkl
│   ├── model_metadata.json
│   ├── classification_report_all.json
│   ├── classification_report_known.json
│   └── confusion_matrix.png
└── hyperparameter_tuning_results.ipynb
```

## Model Performance

### Random Forest Classifier

**Hyperparameters:**
- n_estimators: 213
- max_depth: 27
- min_samples_split: 15
- min_samples_leaf: 6
- max_features: sqrt
- class_weight: balanced

### Results

#### All Test Samples (including novel attacks)

| Metric | Score |
|--------|-------|
| Accuracy | 71.93% |
| Macro F1 | 25.25% |
| Weighted F1 | 61.78% |

*The lower scores here are expected because the model cannot correctly classify novel attack types it was never trained on.*

#### Known Attacks Only (excluding novel attacks)

| Metric | Score |
|--------|-------|
| Accuracy | **86.28%** |
| Macro F1 | 48.85% |
| Weighted F1 | **81.38%** |

### Per-Class Performance (Known Attacks)

| Attack Type | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| normal | 81.61% | 97.27% | 88.76% | 9,711 |
| neptune | 100.00% | 98.52% | 99.25% | 4,657 |
| smurf | 98.96% | 100.00% | 99.48% | 665 |
| satan | 78.90% | 99.73% | 88.10% | 735 |
| ipsweep | 97.87% | 97.87% | 97.87% | 141 |
| nmap | 97.33% | 100.00% | 98.65% | 73 |
| back | 100.00% | 100.00% | 100.00% | 359 |
| portsweep | 70.89% | 96.18% | 81.62% | 157 |
| land | 100.00% | 100.00% | 100.00% | 7 |

### Key Observations

1. **Excellent detection of DoS attacks:** Neptune (99.25% F1), Smurf (99.48% F1), Back (100% F1), and Land (100% F1) are detected with near-perfect accuracy.

2. **Strong probe detection:** Ipsweep (97.87% F1), Nmap (98.65% F1), and Satan (88.10% F1) show strong detection rates.

3. **Challenges with rare attacks:** Low-frequency attacks like buffer_overflow, rootkit, and multihop have poor detection due to limited training samples.

4. **Class imbalance impact:** Despite using balanced class weights, rare attack types with <20 training samples remain difficult to detect.

5. **Novel attack limitation:** The model cannot detect attack types not seen during training, highlighting the need for anomaly-based detection methods for zero-day attacks.

## Dataset References

- [NSL-KDD Dataset on Kaggle](https://www.kaggle.com/datasets/hassan06/nslkdd/data)
