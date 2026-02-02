from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import optuna

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

        valid_params = self.model.get_params()

        for key, value in params.items():
            if key in valid_params:
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

    def tune_hyperparameters(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_val: pd.DataFrame,
            y_val: pd.Series,
            n_trials: int = 50,
            metric: str = 'macro_f1'
    ) -> Dict[str, Any]:
        """
        Tune hyperparameters using Optuna.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            n_trials: Number of optimization trials
            metric: Metric to optimize ('macro_f1', 'weighted_f1', 'accuracy')
        """

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
                'max_depth': trial.suggest_int('max_depth', 10, 50),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                "bootstrap": self.bootstrap,
                'class_weight': self.class_weight,
                'random_state': self.random_state,
                'n_jobs': self.n_jobs
            }

            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            if metric == 'macro_f1':
                score = f1_score(y_val, y_pred, average='macro', zero_division=0)
            elif metric == 'weighted_f1':
                score = f1_score(y_val, y_pred, average='weighted', zero_division=0)
            elif metric == 'accuracy':
                score = accuracy_score(y_val, y_pred)
            else:
                raise ValueError(f"Unknown metric: {metric}")

            return score

        print(f"Starting hyperparameter tuning with {n_trials} trials...")
        print(f"Optimizing for: {metric}")

        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )

        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        print(f"\nBest {metric}: {study.best_value:.4f}")
        print("Best hyperparameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")

        self.set_params(**study.best_params)

        return {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'study': study
        }

class XGBoostModel(BaseModel):
    """
    XGBoost classifier wrapper.
    Requires numeric labels - use LabelEncoderWrapper for y.
    """

    def __init__(
            self,
            n_estimators: int = 100,
            learning_rate: float = 0.1,
            max_depth: int = 6,
            min_child_weight: int = 1,
            subsample: float = 1.0,
            colsample_bytree: float = 1.0,
            gamma: float = 0.0,
            reg_alpha: float = 0.0,
            reg_lambda: float = 1.0,
            random_state: Optional[int] = 42,
            n_jobs: int = -1,
            verbosity: int = 1
    ):
        """
        Initialize XGBoost model.

        Args:
            n_estimators: Number of boosting rounds
            learning_rate: Step size shrinkage
            max_depth: Maximum tree depth
            min_child_weight: Minimum sum of instance weight
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of columns
            gamma: Minimum loss reduction for split
            reg_alpha: L1 regularization
            reg_lambda: L2 regularization
            random_state: Random seed
            n_jobs: Number of parallel jobs
            verbosity: Verbosity level
        """
        super().__init__()

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.gamma = gamma
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbosity = verbosity

        self.sample_weights: Optional[np.ndarray] = None

        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_child_weight=self.min_child_weight,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            gamma=self.gamma,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbosity=self.verbosity,
            eval_metric='mlogloss'
        )

    def fit(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            sample_weight: Optional[np.ndarray] = None
    ) -> 'XGBoostModel':
        """
        Train XGBoost model.

        Args:
            X_train: Training features
            y_train: Training labels (MUST be numeric, use LabelEncoderWrapper)
            sample_weight: Sample weights for imbalanced classes
        """
        print(f"Training XGBoost with {self.n_estimators} estimators...")

        if sample_weight is not None:
            self.sample_weights = sample_weight
            self.model.fit(X_train, y_train, sample_weight=sample_weight)
        else:
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

        if hasattr(self.model, 'feature_names_in_'):
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
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'min_child_weight': self.min_child_weight,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'gamma': self.gamma,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
            'verbosity': self.verbosity
        }

    def set_params(self, **params) -> 'XGBoostModel':
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

        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_child_weight=self.min_child_weight,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            gamma=self.gamma,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbosity=self.verbosity,
            eval_metric='mlogloss'
        )

        self._is_fitted = False

        return self

    def __repr__(self) -> str:
        """String representation."""
        if self._is_fitted:
            return f"XGBoostModel(fitted=True, n_estimators={self.n_estimators})"
        else:
            return f"XGBoostModel(fitted=False, n_estimators={self.n_estimators})"
