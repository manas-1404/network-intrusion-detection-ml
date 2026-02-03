import os
import json
import joblib
import pandas as pd
import numpy as np
import kagglehub
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from data_loader import Dataset
from encoder import OneHotEncoderWrapper, LabelEncoderWrapper
from model import XGBoostModel


def main():
    print("=" * 70)
    print("XGBOOST TRAINING PIPELINE")
    print("=" * 70)

    EVALUATE_KNOWN_ONLY = True

    data_path = kagglehub.dataset_download("hassan06/nslkdd")
    train_path = os.path.join(data_path, "KDDTrain+.txt")
    test_path = os.path.join(data_path, "KDDTest+.txt")

    model_dir = 'xgboost_classifier_model'
    os.makedirs(model_dir, exist_ok=True)

    columns = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
        'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
        'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
        'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
        'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate',
        'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
        'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
        'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
        'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
        'dst_host_srv_rerror_rate', 'label', 'difficulty'
    ]

    print("\n1. Loading Data...")
    dataset = Dataset(train_path, test_path)
    dataset.set_columns(columns)
    dataset.load_data()

    X_train, y_train = dataset.get_X_y(dataset.get_train(), 'label', ['difficulty'])
    X_test, y_test = dataset.get_X_y(dataset.get_test(), 'label', ['difficulty'])

    print("\n2. Encoding Features...")
    categorical_cols = ['protocol_type', 'service', 'flag']

    feature_encoder = OneHotEncoderWrapper(categories='auto')
    feature_encoder.fit(X_train, categorical_cols)

    X_train_encoded = feature_encoder.transform(X_train)
    X_test_encoded = feature_encoder.transform(X_test)

    print(f"   Encoded features: {X_train_encoded.shape[1]}")

    print("\n3. Encoding Labels...")
    label_encoder = LabelEncoderWrapper()
    label_encoder.fit(y_train)

    y_train_encoded = label_encoder.transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    print(f"   Number of classes: {label_encoder.get_num_classes()}")

    print("\n4. Training XGBoost Model...")
    xgb_model = XGBoostModel(
        n_estimators=1017,
        learning_rate=0.1225,
        max_depth=14,
        min_child_weight=3,
        subsample=0.915,
        colsample_bytree=0.980,
        gamma=0.000073,
        reg_alpha=0.112,
        reg_lambda=0.045,
        random_state=355677,
        n_jobs=4,
        verbosity=1
    )

    xgb_model.fit(X_train_encoded, y_train_encoded)

    print("\n5. Evaluating on Test Set...")
    y_pred_encoded = xgb_model.predict(X_test_encoded)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)

    print("\n" + "=" * 70)
    print("EVALUATION ON ALL TEST SAMPLES")
    print("=" * 70)
    print("Description: Performance metrics computed on entire test set,")
    print("including novel attack types that were not present in training.")
    print("Model is penalized for failing to detect unknown attacks.")
    print("=" * 70)

    accuracy_all = accuracy_score(y_test, y_pred)
    macro_f1_all = f1_score(y_test, y_pred, average='macro', zero_division=0)
    weighted_f1_all = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print(f"\nTest Accuracy: {accuracy_all:.4f}")
    print(f"Test Macro F1: {macro_f1_all:.4f}")
    print(f"Test Weighted F1: {weighted_f1_all:.4f}")

    print("\n6. Generating Classification Report (All Samples)...")
    report_all = classification_report(y_test, y_pred, zero_division=0)
    print("\n" + report_all)

    report_all_dict = classification_report(y_test, y_pred, zero_division=0, output_dict=True)

    if EVALUATE_KNOWN_ONLY:
        print("\n" + "=" * 70)
        print("EVALUATION ON KNOWN ATTACKS ONLY")
        print("=" * 70)
        print("Description: Performance metrics computed only on attack types")
        print("that were present in the training data. Novel/unknown attacks")
        print("are excluded from evaluation to assess model performance on")
        print("attacks it was actually trained to detect.")
        print("=" * 70)

        mask_known = y_test_encoded != -1
        num_known = np.sum(mask_known)
        num_unknown = len(y_test) - num_known

        print(f"\nKnown attack samples: {num_known}")
        print(f"Unknown attack samples: {num_unknown}")

        if num_known > 0:
            y_test_known = y_test[mask_known]
            y_pred_known = y_pred[mask_known]

            accuracy_known = accuracy_score(y_test_known, y_pred_known)
            macro_f1_known = f1_score(y_test_known, y_pred_known, average='macro', zero_division=0)
            weighted_f1_known = f1_score(y_test_known, y_pred_known, average='weighted', zero_division=0)

            print(f"\nTest Accuracy (known only): {accuracy_known:.4f}")
            print(f"Test Macro F1 (known only): {macro_f1_known:.4f}")
            print(f"Test Weighted F1 (known only): {weighted_f1_known:.4f}")

            print("\n7. Generating Classification Report (Known Attacks Only)...")
            report_known = classification_report(y_test_known, y_pred_known, zero_division=0)
            print("\n" + report_known)

            report_known_dict = classification_report(y_test_known, y_pred_known, zero_division=0, output_dict=True)
        else:
            print("\nNo known attacks in test set!")
            report_known_dict = None
            accuracy_known = None
            macro_f1_known = None
            weighted_f1_known = None
    else:
        report_known_dict = None
        accuracy_known = None
        macro_f1_known = None
        weighted_f1_known = None

    cm = confusion_matrix(y_test, y_pred)

    print("\n8. Saving Model and Encoders...")

    model_path = os.path.join(model_dir, 'xgb_model.pkl')
    joblib.dump(xgb_model.get_model(), model_path)
    print(f"   Model saved: {model_path}")

    feature_encoder_path = os.path.join(model_dir, 'feature_encoder.pkl')
    joblib.dump(feature_encoder, feature_encoder_path)
    print(f"   Feature encoder saved: {feature_encoder_path}")

    label_encoder_path = os.path.join(model_dir, 'label_encoder.pkl')
    joblib.dump(label_encoder, label_encoder_path)
    print(f"   Label encoder saved: {label_encoder_path}")

    metadata = {
        'model_type': 'XGBClassifier',
        'hyperparameters': {
            'n_estimators': 1017,
            'learning_rate': 0.1225,
            'max_depth': 14,
            'min_child_weight': 3,
            'subsample': 0.915,
            'colsample_bytree': 0.980,
            'gamma': 0.000073,
            'reg_alpha': 0.112,
            'reg_lambda': 0.045,
            'random_state': 355677
        },
        'performance_all_samples': {
            'description': 'Metrics on entire test set including novel attacks',
            'test_accuracy': float(accuracy_all),
            'test_macro_f1': float(macro_f1_all),
            'test_weighted_f1': float(weighted_f1_all)
        },
        'performance_known_only': {
            'description': 'Metrics excluding novel attacks not in training data',
            'test_accuracy': float(accuracy_known) if accuracy_known is not None else None,
            'test_macro_f1': float(macro_f1_known) if macro_f1_known is not None else None,
            'test_weighted_f1': float(weighted_f1_known) if weighted_f1_known is not None else None
        },
        'dataset': {
            'num_train_samples': len(X_train),
            'num_test_samples': len(X_test),
            'num_features': X_train_encoded.shape[1],
            'num_classes': label_encoder.get_num_classes()
        },
        'classes': label_encoder.get_classes().tolist(),
        'categorical_features': categorical_cols,
        'feature_names': feature_encoder.get_encoded_feature_names(),
        'evaluate_known_only_flag': EVALUATE_KNOWN_ONLY
    }

    metadata_path = os.path.join(model_dir, 'model_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   Metadata saved: {metadata_path}")

    report_all_path = os.path.join(model_dir, 'classification_report_all.json')
    with open(report_all_path, 'w') as f:
        json.dump(report_all_dict, f, indent=2)
    print(f"   Classification report (all samples) saved: {report_all_path}")

    if report_known_dict is not None:
        report_known_path = os.path.join(model_dir, 'classification_report_known.json')
        with open(report_known_path, 'w') as f:
            json.dump(report_known_dict, f, indent=2)
        print(f"   Classification report (known only) saved: {report_known_path}")

    cm_path = os.path.join(model_dir, 'confusion_matrix.png')
    class_labels = sorted(set(y_test) | set(y_pred))
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - XGBoost')
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"   Confusion matrix saved: {cm_path}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()