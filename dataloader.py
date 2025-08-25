import pandas as pd
import logging
from pathlib import Path
import requests
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataLoader:

    def __init__(self, url: str, data_dir: str = "data", filename: str = "dataset.csv"):
        self.url = url
        self.data_dir = Path(data_dir)
        self.filename = filename
        self.file_path = self.data_dir / self.filename
        self.scaler = StandardScaler()
        self.label_encoders = {}

        self._ensure_data_directory()
        logger.info(f"DataLoader initialized for {self.file_path}")

    def _ensure_data_directory(self) -> None:
        """Create data directory if it doesn't exist."""
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Data directory ensured: {self.data_dir}")
        except Exception as e:
            logger.error(f"Directory creation error: {e}")
            raise

    def _download_file(self) -> bool:
        """Download file with retry mechanism."""
        for attempt in range(3):
            try:
                logger.info(f"Download attempt {attempt + 1}: {self.url}")

                response = requests.get(self.url, timeout=30, stream=True)
                response.raise_for_status()

                with open(self.file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

                logger.info(f"File successfully downloaded: {self.file_path}")
                return True

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < 2:
                    time.sleep(2)
                else:
                    logger.error("All download attempts failed")
                    return False
        return False

    def load_data(self) -> pd.DataFrame:

        if not self.file_path.exists():
            logger.info("Local file not found, downloading...")
            if not self._download_file():
                raise FileNotFoundError(f"Failed to download from {self.url}")

        try:
            df = pd.read_csv(self.file_path)
            logger.info(f"Data loaded successfully, shape: {df.shape}")
            return df

        except Exception as e:
            logger.error(f"Data loading error: {e}")
            raise

    def preprocess_data(self, df: pd.DataFrame, target_column: str = None) -> pd.DataFrame:

        logger.info("Starting data preprocessing...")

        # Create a copy to avoid modifying original data
        processed_df = df.copy()

        # 1. Cleaning: Handle missing values
        numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
        categorical_cols = processed_df.select_dtypes(include=['object']).columns

        # Fill numeric missing values with median
        for col in numeric_cols:
            if processed_df[col].isnull().any():
                processed_df[col].fillna(processed_df[col].median(), inplace=True)
                logger.info(f"Filled missing values in {col} with median")

        # Fill categorical missing values with mode
        for col in categorical_cols:
            if processed_df[col].isnull().any():
                processed_df[col].fillna(processed_df[col].mode()[0], inplace=True)
                logger.info(f"Filled missing values in {col} with mode")

        # 2. Label encoding for categorical columns (excluding target if specified)
        columns_to_encode = categorical_cols
        if target_column and target_column in categorical_cols:
            columns_to_encode = [col for col in categorical_cols if col != target_column]

        for col in columns_to_encode:
            le = LabelEncoder()
            processed_df[col] = le.fit_transform(processed_df[col].astype(str))
            self.label_encoders[col] = le
            logger.info(f"Label encoded column: {col}")

        # 3. Normalization of numeric columns (excluding target if specified)
        cols_to_normalize = numeric_cols
        if target_column and target_column in numeric_cols:
            cols_to_normalize = [col for col in numeric_cols if col != target_column]

        if len(cols_to_normalize) > 0:
            processed_df[cols_to_normalize] = self.scaler.fit_transform(processed_df[cols_to_normalize])
            logger.info(f"Normalized columns: {cols_to_normalize}")

        logger.info("Data preprocessing completed")
        return processed_df

    def train_test_split_data(self, df: pd.DataFrame, target_column: str,
                              test_size: float = 0.2, random_state: int = 42):
        """
        Split data into train and test sets.

        Args:
            df: Input DataFrame
            target_column: Name of target column
            test_size: Proportion of test data
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info(f"Splitting data with test_size={test_size}")

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")

        X = df.drop(columns=[target_column])
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        return X_train, X_test, y_train, y_test

    def save_csv_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                             y_train: pd.Series, y_test: pd.Series):
        """
        Save structured data to CSV files in data directory.
        """
        logger.info("Saving data...")

        # Save all files in the data directory
        X_train.to_csv(self.data_dir / "X_train.csv", index=False)
        X_test.to_csv(self.data_dir / "X_test.csv", index=False)
        y_train.to_csv(self.data_dir / "y_train.csv", index=False)
        y_test.to_csv(self.data_dir / "y_test.csv", index=False)

        logger.info(f"Structured data saved to {self.data_dir}")
        logger.info(f"Files created: X_train.csv, X_test.csv, y_train.csv, y_test.csv")

    def get_file_path(self) -> Path:
        """Return the path to the data file."""
        return self.file_path


# Example usage
if __name__ == "__main__":
    loader = DataLoader(
        url="https://huggingface.co/datasets/scikit-learn/iris/resolve/main/Iris.csv",
        data_dir="data",
        filename="Iris.csv"
    )

    # Load raw data
    df = loader.load_data()

    # Preprocess data
    processed_df = loader.preprocess_data(df, target_column="Species")

    # Split into train and test
    X_train, X_test, y_train, y_test = loader.train_test_split_data(
        processed_df, target_column="Species", test_size=0.2
    )

    # Save structured data
    loader.save_csv_data(X_train, X_test, y_train, y_test)

    # Display results
    logger.info(f"Original data shape: {df.shape}")
    logger.info(f"Processed data shape: {processed_df.shape}")
    logger.info(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

