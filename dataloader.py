"""Data loading module for Iris dataset."""

import pandas as pd
import os
import logging
from typing import Optional, Tuple
import requests
from pathlib import Path
import time
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration for data loading."""
    url: str = "https://huggingface.co/datasets/scikit-learn/iris/resolve/main/Iris.csv"
    data_dir: str = "data"
    filename: str = "Iris.csv"
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 2.0


class DataLoader:
    """Class for loading and managing dataset."""

    def __init__(self, config: Optional[DataConfig] = None):
        self.config = config or DataConfig()
        self._ensure_data_directory()

    def _ensure_data_directory(self) -> None:
        """Create data directory if it doesn't exist."""
        try:
            Path(self.config.data_dir).mkdir(parents=True, exist_ok=True)
            logger.info(f"Data directory ensured: {self.config.data_dir}")
        except PermissionError:
            logger.error(f"Permission denied to create directory: {self.config.data_dir}")
            raise
        except OSError as e:
            logger.error(f"Failed to create data directory: {e}")
            raise

    def _download_file(self) -> bool:
        """Download the dataset file with retry mechanism."""
        file_path = Path(self.config.data_dir) / self.config.filename

        for attempt in range(self.config.max_retries):
            try:
                logger.info(f"Download attempt {attempt + 1} for {self.config.url}")

                response = requests.get(
                    self.config.url,
                    timeout=self.config.timeout,
                    stream=True
                )
                response.raise_for_status()

                # Save file in chunks to handle large files
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

                logger.info(f"Successfully downloaded file to {file_path}")
                return True

            except requests.exceptions.RequestException as e:
                logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
                else:
                    logger.error(f"All download attempts failed for {self.config.url}")
                    return False
            except IOError as e:
                logger.error(f"File write error: {e}")
                return False

        return False

    def load_data(self) -> Tuple[pd.DataFrame, bool]:
        """
        Load Iris dataset from local file or download if not exists.

        Returns:
            Tuple containing DataFrame and boolean indicating if data was downloaded
        """
        file_path = Path(self.config.data_dir) / self.config.filename
        downloaded = False

        # Check if file exists locally
        if not file_path.exists():
            logger.info("Local file not found, attempting download...")
            success = self._download_file()
            if not success:
                raise FileNotFoundError(f"Failed to download dataset from {self.config.url}")
            downloaded = True

        try:
            # Load data with proper error handling
            df = pd.read_csv(file_path)
            logger.info(f"Successfully loaded dataset with shape {df.shape}")

            # Basic data validation
            if df.empty:
                logger.warning("Loaded dataset is empty")

            return df, downloaded

        except pd.errors.EmptyDataError:
            logger.error("The dataset file is empty")
            raise
        except pd.errors.ParserError:
            logger.error("Error parsing CSV file")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading data: {e}")
            raise

    def get_dataset_info(self, df: pd.DataFrame) -> dict:
        """Get basic information about the dataset."""
        return {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "null_counts": df.isnull().sum().to_dict(),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 ** 2
        }


def main() -> None:
    """Main function to demonstrate data loading."""
    try:
        # Initialize data loader
        config = DataConfig()
        loader = DataLoader(config)

        # Load data
        df, downloaded = loader.load_data()

        # Log dataset information
        info = loader.get_dataset_info(df)
        logger.info(f"Dataset loaded successfully. Info: {info}")

        # Additional processing can be added here
        logger.info(f"First few rows:\n{df.head()}")

    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise


if __name__ == "__main__":
    main()