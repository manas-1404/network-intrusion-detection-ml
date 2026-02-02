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
from model import RandomForestModel


def main():
    print("=" * 70)
    print("RANDOM FOREST TRAINING PIPELINE")
    print("=" * 70)

    data_path = kagglehub.dataset_download("hassan06/nslkdd")
    train_path = os.path.join(data_path, "KDDTrain+.txt")
    test_path = os.path.join(data_path, "KDDTest+.txt")

    model_dir = 'models'
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

    print("\n4. Training Random Forest Model...")
    rf_model = RandomForestModel(
        n_estimators=213,
        criterion='gini',
        max_depth=27,
        min_samples_split=15,
        min_samples_leaf=6,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    rf_model.fit(X_train_encoded, y_train_encoded)

    print("\n5. Evaluating on Test Set...")
    y_pred_encoded = rf_model.predict(X_test_encoded)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print(f"\n   Test Accuracy: {accuracy:.4f}")
    print(f"   Test Macro F1: {macro_f1:.4f}")
    print(f"   Test Weighted F1: {weighted_f1:.4f}")

    print("\n6. Generating Classification Report...")
    report = classification_report(y_test, y_pred, zero_division=0)
    print("\n" + report)

    cm = confusion_matrix(y_test, y_pred)

    print("\n7. Saving Model and Encoders...")

    model_path = os.path.join(model_dir, 'rf_model.pkl')
    joblib.dump(rf_model.get_model(), model_path)
    print(f"   Model saved: {model_path}")

    feature_encoder_path = os.path.join(model_dir, 'feature_encoder.pkl')
    joblib.dump(feature_encoder, feature_encoder_path)
    print(f"   Feature encoder saved: {feature_encoder_path}")

    label_encoder_path = os.path.join(model_dir, 'label_encoder.pkl')
    joblib.dump(label_encoder, label_encoder_path)
    print(f"   Label encoder saved: {label_encoder_path}")

    metadata = {
        'model_type': 'RandomForestClassifier',
        'hyperparameters': {
            'n_estimators': 213,
            'criterion': 'gini',
            'max_depth': 27,
            'min_samples_split': 15,
            'min_samples_leaf': 6,
            'max_features': 'sqrt',
            'class_weight': 'balanced',
            'random_state': 355677
        },
        'performance': {
            'test_accuracy': float(accuracy),
            'test_macro_f1': float(macro_f1),
            'test_weighted_f1': float(weighted_f1)
        },
        'dataset': {
            'num_train_samples': len(X_train),
            'num_test_samples': len(X_test),
            'num_features': X_train_encoded.shape[1],
            'num_classes': label_encoder.get_num_classes()
        },
        'classes': label_encoder.get_classes().tolist(),
        'categorical_features': categorical_cols,
        'feature_names': feature_encoder.get_encoded_feature_names()
    }

    metadata_path = os.path.join(model_dir, 'model_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   Metadata saved: {metadata_path}")

    report_dict = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    report_path = os.path.join(model_dir, 'classification_report.json')
    with open(report_path, 'w') as f:
        json.dump(report_dict, f, indent=2)
    print(f"   Classification report saved: {report_path}")

    cm_path = os.path.join(model_dir, 'confusion_matrix.npy')
    np.save(cm_path, cm)
    print(f"   Confusion matrix saved: {cm_path}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()