import logging
from pathlib import Path
from typing import Tuple, Union

from torch.utils.data import DataLoader

from cxr_detect.data.dataset import CXRDataset
from cxr_detect.data.transforms import (
    get_train_transforms,
    get_val_transforms,
    get_test_transforms,
)
from cxr_detect.utils.config import load_disease_classes
from cxr_detect.utils.process_data import load_data
from cxr_detect.data.encode_labels import encode_labels
from cxr_detect.data.split import split_and_save

LOGGER = logging.getLogger(__name__)


def prepare_data_if_needed(
    raw_csv_path: Union[str, Path], processed_dir: Union[str, Path], seed: int = 42
) -> Path:
    """
    Checks if preprocessed splits exist. If not, runs the encoding
    and patient-aware splitting pipelines automatically.
    """
    raw_csv_path = Path(raw_csv_path)
    processed_dir = Path(processed_dir)

    train_path = processed_dir / "train.csv"
    val_path = processed_dir / "val.csv"
    test_path = processed_dir / "test.csv"

    # 1. Check if processing is already done
    if train_path.exists() and val_path.exists() and test_path.exists():
        LOGGER.info(
            f"Processed splits found in {processed_dir}. Skipping preprocessing."
        )
        return processed_dir

    LOGGER.info("Processed splits not found. Starting automated preprocessing...")

    if not raw_csv_path.exists():
        raise FileNotFoundError(f"Raw dataset CSV not found at {raw_csv_path}")

    # 2. Load raw data
    LOGGER.info("Loading raw data...")
    df = load_data(raw_csv_path)

    # 3. Encode labels
    LOGGER.info("Encoding disease labels...")
    classes = load_disease_classes()
    encoded_df = encode_labels(df, classes)

    # 4. Split by patient ID & save
    # This prevents the same patient from being in train AND val
    LOGGER.info("Splitting dataset by patient ID...")
    processed_dir.mkdir(parents=True, exist_ok=True)
    split_and_save(encoded_df, processed_dir, seed=seed)

    LOGGER.info("Preprocessing complete.")
    return processed_dir


def create_dataloaders(
    raw_csv_path: Union[str, Path],
    processed_dir: Union[str, Path],
    img_dir: Union[str, Path],
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: int = 224,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates train, validation, and test DataLoaders.
    Orchestrates preprocessing if necessary.

    Args:
        raw_csv_path: Path to the original Data_Entry_2017.csv.
        processed_dir: Directory where train.csv, val.csv, and test.csv will be saved/loaded.
        img_dir: Directory containing the actual .png images.
        batch_size: Number of images per batch.
        num_workers: Number of subprocesses for data loading.
        img_size: Target size to resize images to (e.g., 224).
        seed: Random seed for patient splitting.

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    img_dir = Path(img_dir)

    # 1. Run data preparation (encodes and splits if needed)
    processed_path = prepare_data_if_needed(
        raw_csv_path=raw_csv_path, processed_dir=processed_dir, seed=seed
    )

    # 2. Create Datasets mapping to the newly generated split CSVs
    train_dataset = CXRDataset(
        csv_path=processed_path / "train.csv",
        img_dir=img_dir,
        transform=get_train_transforms(img_size),
    )

    val_dataset = CXRDataset(
        csv_path=processed_path / "val.csv",
        img_dir=img_dir,
        transform=get_val_transforms(img_size),
    )

    test_dataset = CXRDataset(
        csv_path=processed_path / "test.csv",
        img_dir=img_dir,
        transform=get_test_transforms(img_size),
    )

    # 3. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Crucial for training
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle validation
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    LOGGER.info(
        f"Data Loaders Ready: Train={len(train_loader)} batches, "
        f"Val={len(val_loader)} batches, Test={len(test_loader)} batches"
    )

    return train_loader, val_loader, test_loader
