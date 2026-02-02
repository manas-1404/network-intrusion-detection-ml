from abc import ABC, abstractmethod
from typing import List, Optional, Any
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

class Encoder(ABC):
    """
    Abstract base class for all encoders.
    """

    @abstractmethod
    def fit(self, X: pd.DataFrame, categorical_cols: List[str]) -> 'Encoder':
        """
        Fit encoder on training data.

        Args:
            X: Feature dataframe
            categorical_cols: List of categorical column names
        """
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted encoder.

        Args:
            X: Feature dataframe
        """
        pass

    def fit_transform(self, X: pd.DataFrame, categorical_cols: List[str]) -> pd.DataFrame:
        """
        Fit and transform in one step.

        Args:
            X: Feature dataframe
            categorical_cols: List of categorical column names
        """
        self.fit(X, categorical_cols)
        return self.transform(X)


class OneHotEncoderWrapper(Encoder):
    """
    One-hot encoder for categorical features.
    """

    def __init__(self, categories: Any, drop: Any = None, sparse_output: bool = False, handle_unknown: str = 'ignore'):
        """
        Initialize one-hot encoder.

        Args:
            sparse_output: Whether to return sparse matrix
            handle_unknown: How to handle unknown categories in transform
        """
        self.sparse_output = sparse_output
        self.handle_unknown = handle_unknown
        self.encoder = OneHotEncoder(
            categories=categories,
            drop=drop,
            sparse_output=self.sparse_output,
            handle_unknown=self.handle_unknown
        )

        self.categorical_cols: Optional[List[str]] = None
        self.numerical_cols: Optional[List[str]] = None
        self.encoded_feature_names: Optional[np.ndarray] = None
        self._numerical_df: Optional[pd.DataFrame] = None
        self._categorical_encoded_df: Optional[pd.DataFrame] = None
        self._combined_df: Optional[pd.DataFrame] = None
        self._is_fitted = False

    def fit(self, X: pd.DataFrame, categorical_cols: List[str]) -> 'OneHotEncoderWrapper':
        """
        Fit one-hot encoder on categorical columns.

        Args:
            X: Feature dataframe
            categorical_cols: List of categorical column names
        """
        missing_cols = set(categorical_cols) - set(X.columns)
        if missing_cols:
            raise ValueError(f"Categorical columns not found in X: {missing_cols}")

        self.categorical_cols = categorical_cols
        self.numerical_cols = [col for col in X.columns if col not in categorical_cols]

        X_categorical = X[self.categorical_cols]

        self.encoder.fit(X_categorical)

        self.encoded_feature_names = self.encoder.get_feature_names_out(self.categorical_cols)

        self._is_fitted = True

        print(f"Fitted encoder on {len(categorical_cols)} categorical columns")
        print(f"Created {len(self.encoded_feature_names)} one-hot encoded features")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform categorical columns to one-hot encoding.
        Caches the results for efficient retrieval.

        Args:
            X: Feature dataframe
        """
        if not self._is_fitted:
            raise ValueError("Encoder not fitted. Call fit() first.")

        # Separate categorical and numerical
        X_categorical = X[self.categorical_cols]
        self._numerical_df = X[self.numerical_cols].copy()

        X_categorical_encoded = self.encoder.transform(X_categorical)

        # Convert to dataframe and cache
        self._categorical_encoded_df = pd.DataFrame(
            X_categorical_encoded,
            columns=self.encoded_feature_names,
            index=X.index
        )

        self._combined_df = pd.concat([self._numerical_df, self._categorical_encoded_df], axis=1)

        return self._combined_df

    def get_encoded_df(self) -> pd.DataFrame:
        """
        Get the cached encoded dataframe (numerical + categorical).
        """
        if self._combined_df is None:
            raise ValueError("No cached dataframe. Call transform() first.")

        return self._combined_df

    def get_numerical_df(self) -> pd.DataFrame:
        """
        Get the cached numerical dataframe.
        """
        if self._numerical_df is None:
            raise ValueError("No cached dataframe. Call transform() first.")

        return self._numerical_df

    def get_categorical_encoded_df(self) -> pd.DataFrame:
        """
        Get the cached one-hot encoded categorical dataframe.
        """
        if self._categorical_encoded_df is None:
            raise ValueError("No cached dataframe. Call transform() first.")

        return self._categorical_encoded_df

    def get_encoded_feature_names(self) -> List[str]:
        """
        Get names of all features after encoding.
        """
        if not self._is_fitted:
            raise ValueError("Encoder not fitted. Call fit() first.")

        return list(self.numerical_cols) + list(self.encoded_feature_names)

    def get_encoded_vectors(self) -> np.ndarray:
        """
        Get encoded feature vectors as numpy array.
        """
        if self._combined_df is None:
            raise ValueError("No cached dataframe. Call transform() first.")

        return self._combined_df.values

    def get_categorical_feature_names(self) -> List[str]:
        """
        Get names of one-hot encoded categorical features only.
        """
        if not self._is_fitted:
            raise ValueError("Encoder not fitted. Call fit() first.")

        return list(self.encoded_feature_names)

    def get_numerical_feature_names(self) -> List[str]:
        """
        Get names of numerical features.
        """
        if not self._is_fitted:
            raise ValueError("Encoder not fitted. Call fit() first.")

        return list(self.numerical_cols)

    def get_feature_count(self) -> dict:
        """
        Get count of features by type.
        """
        if not self._is_fitted:
            raise ValueError("Encoder not fitted. Call fit() first.")

        return {
            'numerical': len(self.numerical_cols),
            'categorical_encoded': len(self.encoded_feature_names),
            'total': len(self.numerical_cols) + len(self.encoded_feature_names)
        }

    def is_fitted(self) -> bool:
        """
        Check if encoder has been fitted.
        """
        return self._is_fitted

    def clear_cache(self) -> None:
        """
        Clear cached dataframes to free memory.
        """
        self._numerical_df = None
        self._categorical_encoded_df = None
        self._combined_df = None

    def __repr__(self) -> str:
        """String representation of encoder."""
        if self._is_fitted:
            return (f"OneHotEncoderWrapper(fitted=True, "
                    f"categorical_cols={len(self.categorical_cols)}, "
                    f"encoded_features={len(self.encoded_feature_names)})")
        else:
            return "OneHotEncoderWrapper(fitted=False)"


class LabelEncoderWrapper():
    """
    Label encoder for encoding target labels (ground truths) (y).
    """

    def __init__(self):
        """
        Initialize label encoder.
        """

        self.encoder = LabelEncoder()
        self._is_fitted = False
        self._classes: Optional[np.ndarray] = None
        self._label_to_number: Optional[dict] = None
        self._number_to_label: Optional[dict] = None

    def fit(self, X: pd.Series) -> 'LabelEncoderWrapper':
        """
        Fit encoder on labels.

        Args:
            X: Label series (y, not features)
        """
        self.encoder.fit(X)
        self._is_fitted = True

        self._classes = self.encoder.classes_

        self._label_to_number = {label: idx for idx, label in enumerate(self._classes)}
        self._number_to_label = {idx: label for idx, label in enumerate(self._classes)}

        print(f"Fitted LabelEncoder on {len(self._classes)} classes")
        print(f"Classes: {list(self._classes)}")

        return self

    def transform(self, X: pd.Series) -> np.ndarray:
        """
        Transform string labels to numeric.

        Args:
            X: Label series (y, not features)
        """
        if not self._is_fitted:
            raise ValueError("LabelEncoder not fitted. Call fit() first.")

        return self.encoder.transform(X)

    def inverse_transform(self, y_encoded: np.ndarray) -> np.ndarray:
        """
        Transform numeric labels back to strings.

        Args:
            y_encoded: Numeric label array
        """
        if not self._is_fitted:
            raise ValueError("LabelEncoder not fitted. Call fit() first.")

        return self.encoder.inverse_transform(y_encoded)

    def label_to_number(self, label: str) -> int:
        """
        Convert single label string to number.

        Args:
            label: Label string
        """
        if not self._is_fitted:
            raise ValueError("LabelEncoder not fitted. Call fit() first.")

        if label not in self._label_to_number:
            raise ValueError(f"Unknown label: {label}")

        return self._label_to_number[label]

    def number_to_label(self, number: int) -> str:
        """
        Convert single number to label string.

        Args:
            number: Numeric label
        """
        if not self._is_fitted:
            raise ValueError("LabelEncoder not fitted. Call fit() first.")

        if number not in self._number_to_label:
            raise ValueError(f"Unknown number: {number}")

        return self._number_to_label[number]

    def labels_to_numbers(self, labels: List[str]) -> List[int]:
        """
        Convert list of label strings to numbers.

        Args:
            labels: List of label strings
        """
        if not self._is_fitted:
            raise ValueError("LabelEncoder not fitted. Call fit() first.")

        return [self._label_to_number[label] for label in labels]

    def numbers_to_labels(self, numbers: List[int]) -> List[str]:
        """
        Convert list of numbers to label strings.

        Args:
            numbers: List of numeric labels
        """
        if not self._is_fitted:
            raise ValueError("LabelEncoder not fitted. Call fit() first.")

        return [self._number_to_label[num] for num in numbers]

    def get_classes(self) -> np.ndarray:
        """
        Get all unique classes.
        """
        if not self._is_fitted:
            raise ValueError("LabelEncoder not fitted. Call fit() first.")

        return self._classes

    def get_label_mapping(self) -> dict:
        """
        Get label to number mapping dictionary.
        """
        if not self._is_fitted:
            raise ValueError("LabelEncoder not fitted. Call fit() first.")

        return self._label_to_number.copy()

    def get_number_mapping(self) -> dict:
        """
        Get number to label mapping dictionary.
        """
        if not self._is_fitted:
            raise ValueError("LabelEncoder not fitted. Call fit() first.")

        return self._number_to_label.copy()

    def get_num_classes(self) -> int:
        """
        Get number of unique classes.
        """
        if not self._is_fitted:
            raise ValueError("LabelEncoder not fitted. Call fit() first.")

        return len(self._classes)

    def is_fitted(self) -> bool:
        """
        Check if encoder has been fitted.
        """
        return self._is_fitted

    def __repr__(self) -> str:
        """String representation of encoder."""
        if self._is_fitted:
            return f"LabelEncoderWrapper(fitted=True, num_classes={len(self._classes)})"
        else:
            return "LabelEncoderWrapper(fitted=False)"