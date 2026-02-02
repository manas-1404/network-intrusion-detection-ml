from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier


class BaseModel(ABC):
    """
    Abstract base class for all ML models.
    """

    def __init__(self):
        """
        Initialize base model.
        """
        self.model = None
        self._is_fitted = False

    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'BaseModel':
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training labels
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Feature dataframe
        """
        pass

    def predict_probabilities(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Feature dataframe
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if not hasattr(self.model, 'predict_proba'):
            raise NotImplementedError("Model does not support predict_proba")

        return self.model.predict_proba(X)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series, zero_division: int = 0) -> Dict[str, Any]:
        """
        Evaluate model performance.

        Args:
            X_test: Test features
            y_test: Test labels
            zero_division: Value to return for zero division in metrics
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        y_pred = self.predict(X_test)

        report = classification_report(y_test, y_pred, zero_division=zero_division, output_dict=True)

        cm = confusion_matrix(y_test, y_pred)

        accuracy = report['accuracy']
        macro_f1 = report['macro avg']['f1-score']
        weighted_f1 = report['weighted avg']['f1-score']

        return {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': y_pred
        }

    def print_evaluation(self, X_test: pd.DataFrame, y_test: pd.Series, zero_division: int = 0) -> None:
        """
        Print evaluation metrics.

        Args:
            X_test: Test features
            y_test: Test labels
            zero_division: Value to return for zero division
        """
        results = self.evaluate(X_test, y_test, zero_division)

        print("=" * 70)
        print("MODEL EVALUATION")
        print("=" * 70)
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Macro F1-Score: {results['macro_f1']:.4f}")
        print(f"Weighted F1-Score: {results['weighted_f1']:.4f}")
        print("\n" + "=" * 70)
        print("CLASSIFICATION REPORT")
        print("=" * 70)
        print(classification_report(y_test, results['predictions'], zero_division=zero_division))

    def is_fitted(self) -> bool:
        """
        Check if model has been fitted.
        """
        return self._is_fitted

    def get_model(self):
        """
        Get the underlying sklearn model.
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        return self.model

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """
        Get model hyperparameters.
        """
        pass

    @abstractmethod
    def set_params(self, **params) -> 'BaseModel':
        """
        Set model hyperparameters.

        Args:
            **params: Hyperparameters to set
        """
        pass


class RandomForestModel(BaseModel):
    """
    Random Forest classifier wrapper.
    """

    def __init__(
            self,
            n_estimators: int = 100,
            criterion: str = 'gini',
            max_depth: Optional[int] = None,
            min_samples_split: int = 2,
            min_samples_leaf: int = 1,
            max_features: str = 'sqrt',
            bootstrap: bool = True,
            class_weight: Optional[str] = None,
            random_state: Optional[int] = 42,
            n_jobs: int = -1,
            verbose: int = 1
    ):
        """
        Initialize Random Forest model.

        Args:
            n_estimators: Number of trees
            criterion: Split quality measure
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples to split node
            min_samples_leaf: Minimum samples in leaf
            max_features: Number of features for best split
            bootstrap: Use bootstrap sampling
            class_weight: Class weights
            random_state: Random seed
            n_jobs: Number of parallel jobs
            verbose: Verbosity level
        """
        super().__init__()

        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=self.verbose
        )

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'RandomForestModel':
        """
        Train Random Forest model.

        Args:
            X_train: Training features
            y_train: Training labels (can be strings)
        """
        print(f"Training Random Forest with {self.n_estimators} trees...")
        self.model.fit(X_train, y_train)
        self._is_fitted = True
        print("Training complete!")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Feature dataframe
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        return self.model.predict(X)

    def get_feature_importance(self) -> pd.Series:
        """
        Get feature importance scores.
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        importances = self.model.feature_importances_

        if isinstance(self.model.feature_names_in_, np.ndarray):
            feature_names = self.model.feature_names_in_
        else:
            feature_names = [f"feature_{i}" for i in range(len(importances))]

        return pd.Series(importances, index=feature_names).sort_values(ascending=False)

    def get_params(self) -> Dict[str, Any]:
        """
        Get model hyperparameters.
        """
        return {
            'n_estimators': self.n_estimators,
            'criterion': self.criterion,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features,
            'bootstrap': self.bootstrap,
            'class_weight': self.class_weight,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
            'verbose': self.verbose
        }

    def set_params(self, **params) -> 'RandomForestModel':
        """
        Set model hyperparameters.

        Args:
            **params: Hyperparameters to set
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")

        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=self.verbose
        )

        self._is_fitted = False

        return self

    def __repr__(self) -> str:
        """String representation."""
        if self._is_fitted:
            return f"RandomForestModel(fitted=True, n_estimators={self.n_estimators})"
        else:
            return f"RandomForestModel(fitted=False, n_estimators={self.n_estimators})"