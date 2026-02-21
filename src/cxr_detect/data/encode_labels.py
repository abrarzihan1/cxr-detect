from typing import Sequence
from cxr_detect.utils.config import load_disease_classes
from cxr_detect.utils.process_data import load_data, save_processed_data

import pandas as pd
import logging
import argparse


REQUIRED_COLUMNS = {
    "Image Index",
    "Patient ID",
    "Finding Labels",
    "Patient Age",
    "View Position",
}


COLUMNS_TO_DROP = [
    "Follow-up #",
    "OriginalImage[Width",
    "Height]",
    "OriginalImagePixelSpacing[x",
    "y]",
]


def encode_labels(df: pd.DataFrame, diseases: Sequence[str]) -> pd.DataFrame:
    """
    Rename columns, encode disease labels into one-hot columns,
    and return a transformed DataFrame.

    Args:
        df: Raw input DataFrame.
        diseases: Ordered list of disease labels to encode.

    Returns:
        Transformed DataFrame with one-hot encoded disease columns.

    Raises:
        ValueError: If required columns are missing.
    """
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df_out = df.copy()
    df_out = df_out.rename(
        columns={
            "Image Index": "image_id",
            "Patient ID": "patient_id",
            "Patient Age": "patient_age",
            "Patient Gender": "patient_gender",
            "View Position": "view_position",
        }
    )

    df_out = df_out.drop(
        columns=[c for c in COLUMNS_TO_DROP if c in df_out.columns],
        errors="ignore",
    )

    df_out["patient_age"] = (
        df_out["patient_age"].str.replace("Y", "", regex=False).astype(int)
    )

    # One-hot encode disease labels
    encoded = (
        df_out["Finding Labels"]
        .fillna("")
        .str.get_dummies(sep="|")
        .reindex(columns=diseases, fill_value=0)
    )

    df_out = df_out.drop(columns=["Finding Labels"]).join(encoded)

    return df_out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file", type=str, help="Path to input file", required=True
    )
    parser.add_argument(
        "--output_file", type=str, help="Path to output file", required=True
    )
    args = parser.parse_args()

    logger.info("Loading data...")
    df = load_data(args.input_file)
    logger.info("Encoding labels...")

    classes = load_disease_classes()

    df = encode_labels(df, classes)
    logger.info("Saving processed data...")
    save_processed_data(df, args.output_file)
    logger.info("Done.")
