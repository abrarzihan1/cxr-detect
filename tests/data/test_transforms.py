import pytest
import torch
import numpy as np
from PIL import Image

from cxr_detect.data.transforms import (
    get_train_transforms,
    get_val_transforms,
)


@pytest.fixture
def sample_image():
    """Creates a dummy 224x224 RGB image."""
    # Using a non-black image to test normalization effectively
    # Create a random noise image
    arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def test_train_transforms_structure(sample_image):
    """Test if train transforms return a tensor of correct shape."""
    transform = get_train_transforms(img_size=224)
    output = transform(sample_image)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (3, 224, 224)
    assert output.dtype == torch.float32


def test_val_transforms_structure(sample_image):
    """Test if validation transforms return a tensor of correct shape."""
    transform = get_val_transforms(img_size=224)
    output = transform(sample_image)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (3, 224, 224)


def test_val_transforms_determinism(sample_image):
    """Validation transforms must be deterministic (no random augmentation)."""
    transform = get_val_transforms(img_size=224)

    # Run the transform twice on the same image
    out1 = transform(sample_image)
    out2 = transform(sample_image)

    # The outputs should be exactly identical
    assert torch.equal(out1, out2)


def test_train_transforms_variation(sample_image):
    """
    Train transforms should produce different outputs for the same image
    due to random augmentations.
    Note: There is a tiny chance this fails if random params are identical,
    but it's statistically negligible.
    """
    transform = get_train_transforms(img_size=224)

    out1 = transform(sample_image)
    out2 = transform(sample_image)

    # The outputs should NOT be identical
    assert not torch.equal(out1, out2)


def test_custom_image_size(sample_image):
    """Test if the transforms respect the img_size argument."""
    size = 128
    # We need a larger image to crop down from, or resize up to
    large_image = sample_image.resize((256, 256))

    transform = get_train_transforms(img_size=size)
    output = transform(large_image)

    assert output.shape == (3, size, size)


def test_normalization_values(sample_image):
    """
    Test if values are roughly standardized.
    Unnormalized images are [0, 1]. Normalized should have negative values.
    """
    transform = get_val_transforms(img_size=224)
    output = transform(sample_image)

    # Check that we have values less than 0 (impossible without normalization)
    assert torch.min(output) < 0
    # Check that mean is somewhat close to 0 (given random noise input)
    # It won't be exactly 0 because our random noise isn't real world data
    assert -3.0 < torch.mean(output) < 3.0
