import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_data(path: str | Path) -> pd.DataFrame | None:
    """
    Load a CSV file into a pandas DataFrame.

    Args:
        path: Path to the CSV file.

    Returns:
        A pandas DataFrame containing the file data.

    Raises:
        FileNotFoundError: If the file does not exist.
        pd.errors.ParserError: If the file cannot be parsed.
        OSError: If there is an issue accessing the file.
    """

    file_path = Path(path)

    if not file_path.exists():
        logging.error(f"File {file_path} does not exist.")
        raise FileNotFoundError(f"File {file_path} does not exist.")

    try:
        df = pd.read_csv(file_path)
        logger.info(f"File {file_path} successfully loaded.")
        return df
    except Exception:
        logging.exception(f"File {path} not found")
        raise


def save_processed_data(df: pd.DataFrame, output_path: str | Path) -> None:
    """
    Save a processed DataFrame to a CSV file.

    Args:
        df: The processed pandas DataFrame.
        output_path: Destination file path.

    Raises:
        ValueError: If the DataFrame is empty.
        OSError: If the file cannot be written.
    """
    if df.empty:
        logger.warning("Attempted to save an empty DataFrame.")
        raise ValueError("DataFrame is empty")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        df.to_csv(path, index=False)
        logger.info("Data successfully saved to %s", path)
    except OSError as exc:
        logger.error("Failed to save file: %s", exc)
        raise
