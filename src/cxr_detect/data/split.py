import argparse
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from cxr_detect.utils.process_data import load_data, save_processed_data


LOGGER = logging.getLogger(__name__)
PATIENT_ID_COLUMN = "patient_id"


# ---------------------------------------------------------------------
# Core Logic
# ---------------------------------------------------------------------


def get_unique_patient_ids(df: pd.DataFrame) -> np.ndarray:
    """Return unique patient IDs from dataframe."""
    if PATIENT_ID_COLUMN not in df.columns:
        raise KeyError(f"Column '{PATIENT_ID_COLUMN}' not found in DataFrame.")
    return df[PATIENT_ID_COLUMN].unique()


def split_patient_ids(
    patient_ids: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split patient IDs into train, validation, and test sets."""
    rng = np.random.default_rng(seed)
    shuffled_ids = rng.permutation(patient_ids)

    n_total = len(shuffled_ids)
    train_end = int(n_total * train_ratio)
    val_end = train_end + int(n_total * val_ratio)

    train_ids = shuffled_ids[:train_end]
    val_ids = shuffled_ids[train_end:val_end]
    test_ids = shuffled_ids[val_end:]

    return train_ids, val_ids, test_ids


def filter_by_patient_ids(df: pd.DataFrame, patient_ids: np.ndarray) -> pd.DataFrame:
    """Filter dataframe by patient IDs."""
    return df[df[PATIENT_ID_COLUMN].isin(patient_ids)].copy()


# ---------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------


def save_split(
    df: pd.DataFrame,
    patient_ids: np.ndarray,
    split_name: str,
    output_dir: Path,
) -> None:
    """Save a single split to CSV."""
    split_df = filter_by_patient_ids(df, patient_ids)

    if split_df.empty:
        raise ValueError(f"{split_name} split is empty.")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{split_name}.csv"

    save_processed_data(split_df, output_file)
    LOGGER.info(
        "%s split saved: %d rows (%d patients)",
        split_name,
        len(split_df),
        len(patient_ids),
    )


def split_and_save(
    df: pd.DataFrame,
    output_dir: Path,
    seed: int = 42,
) -> None:
    """Split dataset by patient and save train/val/test files."""
    patient_ids = get_unique_patient_ids(df)

    train_ids, val_ids, test_ids = split_patient_ids(patient_ids, seed=seed)

    save_split(df, train_ids, "train", output_dir)
    save_split(df, val_ids, "val", output_dir)
    save_split(df, test_ids, "test", output_dir)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Split dataset by patient ID.")
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to processed dataset CSV file.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Directory to save split CSV files.",
    )
    return parser.parse_args()


def main() -> None:
    """Main execution entry point."""
    logging.basicConfig(level=logging.INFO)

    args = parse_args()

    LOGGER.info("Loading data from %s", args.input_file)
    df = load_data(args.input_file)

    split_and_save(df, Path(args.output_path))

    LOGGER.info("Data successfully split and saved.")


if __name__ == "__main__":
    main()
