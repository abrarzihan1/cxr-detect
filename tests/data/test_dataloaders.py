import pytest
from unittest.mock import patch
from torch.utils.data import DataLoader, Dataset
import torch
from cxr_detect.data.dataloader import prepare_data_if_needed, create_dataloaders


# --- FIXTURES ---


@pytest.fixture
def mock_dirs(tmp_path):
    """Provides temporary directories for raw and processed data."""
    raw_csv = tmp_path / "raw" / "Data_Entry_2017.csv"
    raw_csv.parent.mkdir(parents=True, exist_ok=True)

    processed_dir = tmp_path / "processed"
    img_dir = tmp_path / "images"

    return raw_csv, processed_dir, img_dir


# A dummy dataset to replace CXRDataset during DataLoader tests
# This prevents the dataloader from looking for real images on disk
class DummyDataset(Dataset):
    def __init__(self, *args, **kwargs):
        pass

    def __len__(self):
        return 10

    def __getitem__(self, idx):
        return torch.zeros(3, 224, 224), torch.zeros(14)


# --- TESTS FOR prepare_data_if_needed ---


@patch("cxr_detect.data.dataloader.split_and_save")
@patch("cxr_detect.data.dataloader.encode_labels")
@patch("cxr_detect.data.dataloader.load_disease_classes")
@patch("cxr_detect.data.dataloader.load_data")
def test_prepare_data_already_exists(
    mock_load_data, mock_load_classes, mock_encode, mock_split, mock_dirs
):
    """If train, val, and test.csv exist, preprocessing should be skipped."""
    raw_csv, processed_dir, _ = mock_dirs

    # Create the fake processed files
    processed_dir.mkdir(parents=True, exist_ok=True)
    (processed_dir / "train.csv").touch()
    (processed_dir / "val.csv").touch()
    (processed_dir / "test.csv").touch()

    result_path = prepare_data_if_needed(raw_csv, processed_dir)

    assert result_path == processed_dir
    # Ensure none of the heavy processing functions were called
    mock_load_data.assert_not_called()
    mock_encode.assert_not_called()
    mock_split.assert_not_called()


@patch("cxr_detect.data.dataloader.split_and_save")
@patch("cxr_detect.data.dataloader.encode_labels")
@patch("cxr_detect.data.dataloader.load_disease_classes")
@patch("cxr_detect.data.dataloader.load_data")
def test_prepare_data_runs_pipeline(
    mock_load_data, mock_load_classes, mock_encode, mock_split, mock_dirs
):
    """If processed files are missing, the pipeline should run."""
    raw_csv, processed_dir, _ = mock_dirs

    # Create the fake raw CSV
    raw_csv.touch()

    # Setup mock returns
    mock_load_data.return_value = "dummy_raw_df"
    mock_load_classes.return_value = ["DiseaseA", "DiseaseB"]
    mock_encode.return_value = "dummy_encoded_df"

    result_path = prepare_data_if_needed(raw_csv, processed_dir, seed=99)

    assert result_path == processed_dir

    # Verify the pipeline was executed in the correct order
    mock_load_data.assert_called_once_with(raw_csv)
    mock_load_classes.assert_called_once()
    mock_encode.assert_called_once_with("dummy_raw_df", ["DiseaseA", "DiseaseB"])
    mock_split.assert_called_once_with("dummy_encoded_df", processed_dir, seed=99)


def test_prepare_data_missing_raw(mock_dirs):
    """If raw CSV doesn't exist, it should raise a FileNotFoundError."""
    raw_csv, processed_dir, _ = mock_dirs
    # We purposefully DO NOT touch/create the raw_csv

    with pytest.raises(FileNotFoundError):
        prepare_data_if_needed(raw_csv, processed_dir)


# --- TESTS FOR create_dataloaders ---


@patch("cxr_detect.data.dataloader.CXRDataset", new=DummyDataset)
@patch("cxr_detect.data.dataloader.prepare_data_if_needed")
def test_create_dataloaders(mock_prepare, mock_dirs):
    """Test if dataloaders are instantiated with the correct parameters."""
    raw_csv, processed_dir, img_dir = mock_dirs

    # Mock prepare_data to just return the processed dir path
    mock_prepare.return_value = processed_dir

    batch_size = 4
    num_workers = 2

    train_loader, val_loader, test_loader = create_dataloaders(
        raw_csv_path=raw_csv,
        processed_dir=processed_dir,
        img_dir=img_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        img_size=224,
        seed=42,
    )

    # Check return types
    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)
    assert isinstance(test_loader, DataLoader)

    # Check batch sizes
    assert train_loader.batch_size == batch_size
    assert val_loader.batch_size == batch_size
    assert test_loader.batch_size == batch_size

    # Check num_workers
    assert train_loader.num_workers == num_workers

    # Verify Shuffle Logic
    # train_loader.sampler is a RandomSampler if shuffle=True
    from torch.utils.data.sampler import RandomSampler, SequentialSampler

    assert isinstance(train_loader.sampler, RandomSampler)
    assert isinstance(val_loader.sampler, SequentialSampler)
    assert isinstance(test_loader.sampler, SequentialSampler)
