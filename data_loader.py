import pandas as pd
from typing import List, Tuple, Optional

class Dataset:
    """
    Handles loading and basic management of dataset.
    """

    def __init__(self, train_path: str, test_path: str):
        """
        Initialize Dataset with file paths.

        Args:
            train_path: Path to training data file
            test_path: Path to test data file
        """
        self.train_path = train_path
        self.test_path = test_path
        self.columns: Optional[List[str]] = None
        self.train_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None

    def set_columns(self, columns: List[str]) -> None:
        """
        Set column names for the dataset.

        Args:
            columns: List of column names
        """
        if not columns:
            raise ValueError("Columns list cannot be empty")
        self.columns = columns

    def load_data(self) -> None:
        """
        Load training and test data from files.
        """
        if self.columns is None:
            raise ValueError("Must set columns before loading data. Call set_columns() first.")

        try:
            self.train_df = pd.read_csv(
                self.train_path,
                names=self.columns,
                header=None
            )
            self.test_df = pd.read_csv(
                self.test_path,
                names=self.columns,
                header=None
            )
            print(f"Loaded training data: {self.train_df.shape}")
            print(f"Loaded test data: {self.test_df.shape}")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Data file not found: {e}")

    def get_train(self) -> pd.DataFrame:
        """
        Get training dataframe.
        """
        if self.train_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        return self.train_df

    def get_test(self) -> pd.DataFrame:
        """
        Get test dataframe.
        """
        if self.test_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        return self.test_df

    def get_X_y(
            self,
            df: pd.DataFrame,
            label_column: str,
            drop_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separate features (X) and labels (y) from dataframe.

        Args:
            df: Input dataframe
            label_column: Name of label column
            drop_cols: Additional columns to drop besides label
        """
        if drop_cols is None:
            drop_cols = []

        cols_to_drop = [label_column] + drop_cols

        X = df.drop(columns=cols_to_drop)
        y = df[label_column]

        return X, y

    def get_class_distribution(self, series: pd.Series) -> pd.Series:
        """
        Get class distribution for a label series.
        """
        return series.value_counts()

    def __repr__(self) -> str:
        """String representation of Dataset."""
        train_shape = self.train_df.shape if self.train_df is not None else "Not loaded"
        test_shape = self.test_df.shape if self.test_df is not None else "Not loaded"
        return f"Dataset(train={train_shape}, test={test_shape})"