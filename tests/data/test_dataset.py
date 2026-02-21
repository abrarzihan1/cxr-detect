import pytest
import pandas as pd
import torch
from PIL import Image
from unittest.mock import patch

from cxr_detect.data.dataset import CXRDataset


# --- Fixtures: Setup Fake Data ---


@pytest.fixture
def mock_disease_classes():
    """Defines a subset of diseases for testing."""
    return ["Atelectasis", "Cardiomegaly", "Effusion"]


@pytest.fixture
def mock_csv_data(tmp_path, mock_disease_classes):
    """Creates a temporary CSV file with dummy annotations."""
    csv_path = tmp_path / "train.csv"

    # Create a dataframe that matches your structure
    data = {
        "image_id": ["img_01.png", "img_02.png", "img_03.png"],
        "patient_id": [1, 2, 3],
        "patient_age": [50, 60, 70],
        # All zeros initially
        **{disease: [0, 0, 0] for disease in mock_disease_classes},
    }

    df = pd.DataFrame(data)

    # Let's set specific labels to verify later
    # img_01 has Atelectasis (index 0)
    df.loc[0, "Atelectasis"] = 1
    # img_02 has Cardiomegaly (index 1) and Effusion (index 2)
    df.loc[1, "Cardiomegaly"] = 1
    df.loc[1, "Effusion"] = 1

    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def mock_img_dir(tmp_path):
    """Creates temporary dummy images."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    # Create 3 dummy images (224x224 black images)
    for i in range(1, 4):
        img_name = f"img_0{i}.png"
        img = Image.new("RGB", (224, 224), color="black")
        img.save(img_dir / img_name)

    return img_dir


# --- Tests ---


@patch("cxr_detect.data.dataset.load_disease_classes")
def test_dataset_length(mock_load, mock_csv_data, mock_img_dir, mock_disease_classes):
    """Test if len(dataset) returns the correct count."""
    # Configure the mock to return our test classes
    mock_load.return_value = mock_disease_classes

    dataset = CXRDataset(csv_path=mock_csv_data, img_dir=mock_img_dir)

    # We created 3 rows in the CSV
    assert len(dataset) == 3


@patch("cxr_detect.data.dataset.load_disease_classes")
def test_getitem_structure(
    mock_load, mock_csv_data, mock_img_dir, mock_disease_classes
):
    """Test the output types and shapes of __getitem__."""
    mock_load.return_value = mock_disease_classes

    dataset = CXRDataset(csv_path=mock_csv_data, img_dir=mock_img_dir)

    image, label = dataset[0]

    # Check Image
    assert isinstance(image, Image.Image)  # Should be PIL image if no transform
    assert image.size == (224, 224)

    # Check Label
    assert isinstance(label, torch.Tensor)
    assert label.shape == (len(mock_disease_classes),)  # Should be (3,)
    assert label.dtype == torch.float32  # Essential for BCEWithLogitsLoss


@patch("cxr_detect.data.dataset.load_disease_classes")
def test_label_correctness(
    mock_load, mock_csv_data, mock_img_dir, mock_disease_classes
):
    """Test if the specific labels match the CSV data."""
    mock_load.return_value = mock_disease_classes

    dataset = CXRDataset(csv_path=mock_csv_data, img_dir=mock_img_dir)

    # Test Item 0: Atelectasis=1 (Index 0)
    _, label_0 = dataset[0]
    assert torch.equal(label_0, torch.tensor([1.0, 0.0, 0.0]))

    # Test Item 1: Cardiomegaly=1, Effusion=1 (Indices 1 and 2)
    _, label_1 = dataset[1]
    assert torch.equal(label_1, torch.tensor([0.0, 1.0, 1.0]))


@patch("cxr_detect.data.dataset.load_disease_classes")
def test_transforms_applied(
    mock_load, mock_csv_data, mock_img_dir, mock_disease_classes
):
    """Test if transformations are correctly applied."""
    mock_load.return_value = mock_disease_classes

    # specific transform to resize to 100x100
    from torchvision import transforms

    my_transform = transforms.Compose(
        [transforms.Resize((100, 100)), transforms.ToTensor()]
    )

    dataset = CXRDataset(
        csv_path=mock_csv_data, img_dir=mock_img_dir, transform=my_transform
    )

    image, _ = dataset[0]

    # Output should now be a Tensor, not PIL
    assert isinstance(image, torch.Tensor)
    # Output shape should be (Channels, H, W) -> (3, 100, 100)
    assert image.shape == (3, 100, 100)


@patch("cxr_detect.data.dataset.load_disease_classes")
def test_missing_image_error(
    mock_load, mock_csv_data, mock_img_dir, mock_disease_classes
):
    """Test if the dataset raises an error when an image is missing."""
    mock_load.return_value = mock_disease_classes

    # Delete one image from the directory to simulate a missing file
    (mock_img_dir / "img_01.png").unlink()

    dataset = CXRDataset(csv_path=mock_csv_data, img_dir=mock_img_dir)

    # Accessing index 0 (which points to img_01.png) should crash
    with pytest.raises(FileNotFoundError):
        _ = dataset[0]
