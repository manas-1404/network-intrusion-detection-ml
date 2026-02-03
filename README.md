# Network Intrusion Detection using Machine Learning

A machine learning system for detecting network intrusions using the NSL-KDD dataset. The model classifies network traffic as either normal or one of multiple attack types.

## Overview

This project implements a network intrusion detection system (NIDS) using Random Forest and XGBoost classifiers. It's designed to identify malicious network activity by analyzing connection-level features extracted from network traffic.

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

**Note:** The test set contains 17 novel attack types not present in training data (apache2, httptunnel, mailbomb, mscan, named, processtable, ps, saint, sendmail, snmpgetattack, snmpguess, sqlattack, udpstorm, worm, xlock, xsnoop, xterm).

## Installation

```bash
# Clone the repository
git clone https://github.com/manas-1404/ML-Network-Detection.git
cd ML-Network-Detection

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

## Usage

### Training the Models

**Random Forest:**
```bash
python train_random_forest_classifier.py
```

**XGBoost:**
```bash
python train_xgboost_classifier.py
```

Both scripts will:
1. Download the NSL-KDD dataset via kagglehub
2. Preprocess and encode features
3. Train the classifier
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
├── data_loader.py                     # Dataset loading and preprocessing
├── encoder.py                         # Feature and label encoding wrappers
├── model.py                           # Model wrappers (RandomForest, XGBoost)
├── train_random_forest_classifier.py  # Random Forest training pipeline
├── train_xgboost_classifier.py        # XGBoost training pipeline
├── random_forest_classifier_model/    # Random Forest model artifacts
│   ├── rf_model.pkl
│   ├── feature_encoder.pkl
│   ├── label_encoder.pkl
│   ├── model_metadata.json
│   ├── classification_report_all.json
│   ├── classification_report_known.json
│   └── confusion_matrix.png
├── xgboost_classifier_model/          # XGBoost model artifacts
│   ├── xgb_model.pkl
│   ├── feature_encoder.pkl
│   ├── label_encoder.pkl
│   ├── model_metadata.json
│   ├── classification_report_all.json
│   ├── classification_report_known.json
│   └── confusion_matrix.png
└── hyperparameter_tuning_results.ipynb
```

## Hyperparameter Optimization

Both models were optimized using **Optuna** with Bayesian optimization (Tree-structured Parzen Estimator):

- **Random Forest:** 50 trials, optimizing macro F1-score
- **XGBoost:** 50+ trials, optimizing macro F1-score

See `hyperparameter_tuning_results.ipynb` for detailed tuning experiments.

## Model Performance

### Model Comparison Summary

| Model | Known Accuracy | Known Weighted F1 | All Accuracy | All Weighted F1 |
|-------|---------------|-------------------|--------------|-----------------|
| **XGBoost** | **86.59%** | **81.72%** | **72.19%** | **62.18%** |
| Random Forest | 86.28% | 81.38% | 71.93% | 61.78% |

**Note:** "Known Attacks Only" metrics exclude the 17 novel attack types in the test set that were not present during training. This provides a fair evaluation of model performance on attacks it was trained to detect. "All" metrics include novel attacks where the model is expected to fail.

---

### XGBoost Classifier

**Hyperparameters (Optuna-tuned):**
- n_estimators: 1017
- learning_rate: 0.1225
- max_depth: 14
- min_child_weight: 3
- subsample: 0.915
- colsample_bytree: 0.98
- gamma: 0.000073
- reg_alpha: 0.112
- reg_lambda: 0.045

#### All Test Samples (including novel attacks)

| Metric | Score |
|--------|-------|
| Accuracy | 72.19% |
| Macro F1 | 25.28% |
| Weighted F1 | 62.18% |

#### Known Attacks Only

| Metric | Score |
|--------|-------|
| Accuracy | **86.59%** |
| Macro F1 | 49.02% |
| Weighted F1 | **81.72%** |

#### Per-Class Performance (XGBoost)

| Attack Type | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| normal | 81.83% | 97.39% | 88.94% | 9,711 |
| neptune | 100.00% | 100.00% | 100.00% | 4,657 |
| smurf | 97.94% | 100.00% | 98.96% | 665 |
| satan | 81.56% | 99.86% | 89.79% | 735 |
| ipsweep | 97.89% | 98.58% | 98.23% | 141 |
| nmap | 98.63% | 98.63% | 98.63% | 73 |
| back | 100.00% | 100.00% | 100.00% | 359 |
| portsweep | 78.19% | 93.63% | 85.22% | 157 |
| pod | 69.81% | 90.24% | 78.72% | 41 |

---

### Random Forest Classifier

**Hyperparameters (Optuna-tuned):**
- n_estimators: 213
- max_depth: 27
- min_samples_split: 15
- min_samples_leaf: 6
- max_features: sqrt
- class_weight: balanced

#### All Test Samples (including novel attacks)

| Metric | Score |
|--------|-------|
| Accuracy | 71.93% |
| Macro F1 | 25.25% |
| Weighted F1 | 61.78% |

*The lower scores here are expected because the model cannot correctly classify novel attack types it was never trained on.*

#### Known Attacks Only

| Metric | Score |
|--------|-------|
| Accuracy | **86.28%** |
| Macro F1 | 48.85% |
| Weighted F1 | **81.38%** |

#### Per-Class Performance (Random Forest)

| Attack Type | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| normal | 81.61% | 97.27% | 88.76% | 9,711 |
| neptune | 100.00% | 99.00% | 99.00% | 4,657 |
| smurf | 98.96% | 100.00% | 99.48% | 665 |
| satan | 78.90% | 99.73% | 88.10% | 735 |
| ipsweep | 97.87% | 97.87% | 97.87% | 141 |
| nmap | 97.33% | 100.00% | 98.65% | 73 |
| back | 100.00% | 100.00% | 100.00% | 359 |
| portsweep | 70.89% | 96.18% | 81.62% | 157 |
| land | 100.00% | 100.00% | 100.00% | 7 |

---

### Key Observations

1. **Excellent detection of DoS attacks:** Neptune (100% F1 XGBoost), Smurf, and Back are detected with near-perfect accuracy by both models.

2. **Strong probe detection:** Ipsweep (98.23% F1), Nmap (98.63% F1), and Satan (89.79% F1) show strong detection rates.

3. **XGBoost improvements:** Better detection on neptune (+1% F1) and portsweep (+3.6% F1) compared to Random Forest.

4. **Challenges with rare attacks:** Low-frequency attacks like buffer_overflow, rootkit, and multihop have poor detection due to limited training samples.

5. **Class imbalance impact:** Despite using balanced class weights (RF) and tuned hyperparameters, rare attack types with <20 training samples remain difficult to detect.

6. **Novel attack limitation:** Both models cannot detect attack types not seen during training, highlighting the need for anomaly-based detection methods for zero-day attacks.

## References

- [NSL-KDD Dataset on Kaggle](https://www.kaggle.com/datasets/hassan06/nslkdd/data)
