import pytest
import torch
import torch.nn as nn

# Adjust import based on your actual structure
from cxr_detect.models.model_factory import create_model

# --- FIXTURES & PARAMETERS ---

# Test across the supported architectures
SUPPORTED_MODELS = ["resnet18", "resnet34", "resnet50"]
NUM_CLASSES = 14
BATCH_SIZE = 2


@pytest.fixture
def dummy_input():
    """Returns a dummy batch of 2 images with shape (3, 224, 224)."""
    return torch.randn(BATCH_SIZE, 3, 224, 224)


# --- TESTS ---


@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_forward_pass_and_output_shape(model_name, dummy_input):
    """
    Test that the model runs a forward pass without crashing
    and returns logits of the correct shape (Batch, Num_Classes).
    """
    model = create_model(model_name, num_classes=NUM_CLASSES, pretrained=False)

    # Set to eval mode to prevent BatchNorm from expecting large batches
    model.eval()

    with torch.no_grad():
        output = model(dummy_input)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (BATCH_SIZE, NUM_CLASSES)

    # Ensure there is no Sigmoid applied (values shouldn't be strictly bounded to [0,1])
    # With random weights, it's highly likely to have values outside [0,1].
    assert output.min() < 0 or output.max() > 1


def test_model_device_movement():
    """Test that the model parameters can be moved to the correct device."""
    model = create_model("resnet18", num_classes=NUM_CLASSES, pretrained=False)

    # Check initial device (should be CPU by default)
    initial_device = next(model.parameters()).device
    assert initial_device.type == "cpu"

    # Test explicitly moving to CPU
    model = model.to(torch.device("cpu"))
    assert next(model.parameters()).device.type == "cpu"

    # If a GPU is available, test moving to GPU
    if torch.cuda.is_available():
        model = model.to(torch.device("cuda"))
        assert next(model.parameters()).device.type == "cuda"

        # Move dummy input and test forward pass on GPU
        dummy_gpu = torch.randn(BATCH_SIZE, 3, 224, 224).to("cuda")
        output = model(dummy_gpu)
        assert output.device.type == "cuda"


@pytest.mark.parametrize(
    "model_name, min_params",
    [
        ("resnet18", 11_000_000),  # ~11.1M params
        ("resnet34", 21_000_000),  # ~21.2M params
        ("resnet50", 23_000_000),  # ~23.5M params
    ],
)
def test_parameter_count_and_fc_layer(model_name, min_params):
    """
    Sanity check to ensure the backbone is loaded correctly with the right
    number of parameters and the final FC layer is replaced.
    """
    model = create_model(model_name, num_classes=NUM_CLASSES, pretrained=False)

    # Check total parameter count
    total_params = sum(p.numel() for p in model.parameters())
    assert (
        total_params > min_params
    ), f"{model_name} should have at least {min_params} parameters"

    # Verify the fully connected layer was correctly modified
    assert hasattr(model.backbone, "fc")
    assert isinstance(model.backbone.fc, nn.Linear)
    assert model.backbone.fc.out_features == NUM_CLASSES

    # Ensure gradients are enabled for the new layer
    assert model.backbone.fc.weight.requires_grad is True


def test_factory_invalid_model_name():
    """Test that the factory raises an error for unsupported models."""
    with pytest.raises(ValueError, match="is not supported"):
        create_model("invalid_architecture", num_classes=NUM_CLASSES, pretrained=False)
