import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import GradScaler
from copy import deepcopy

# Adjust these imports based on your actual structure
from cxr_detect.training.train_one_epoch import train_one_epoch
from cxr_detect.training.evaluate import evaluate


# --- FIXTURES ---


@pytest.fixture
def device():
    # We use CPU for unit tests to ensure they run anywhere
    return torch.device("cpu")


@pytest.fixture
def scaler():
    # For CPU tests, it is best practice to disable the scaler
    # to avoid warnings, as AMP is designed for CUDA GPUs.
    # In your actual training script (Trainer), it will default to enabled=True.
    return GradScaler(enabled=False)


@pytest.fixture
def num_classes():
    return 14


@pytest.fixture
def dummy_loader(num_classes):
    """Creates a small DataLoader with dummy images and binary targets."""
    batch_size = 4
    num_samples = 10

    # We use small 16x16 images to keep tests lightning fast
    x = torch.randn(num_samples, 3, 16, 16)
    y = torch.randint(0, 2, (num_samples, num_classes)).float()

    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


@pytest.fixture
def dummy_model(num_classes):
    """Creates a tiny valid PyTorch model that expects (B, 3, 16, 16) inputs."""
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 16 * 16, num_classes))
    return model


@pytest.fixture
def criterion():
    return nn.BCEWithLogitsLoss()


@pytest.fixture
def optimizer(dummy_model):
    # High learning rate so weight changes are obvious in tests
    return torch.optim.SGD(dummy_model.parameters(), lr=0.1)


# --- TESTS ---


def test_train_one_epoch_updates_weights(
    dummy_model, dummy_loader, criterion, optimizer, scaler, device
):
    """
    Test that train_one_epoch successfully runs and actually updates model weights.
    """
    # 1. Save a deep copy of the initial weights to compare later
    initial_weights = deepcopy(list(dummy_model.parameters())[0].data)

    # 2. Run training
    epoch_loss = train_one_epoch(
        model=dummy_model,
        dataloader=dummy_loader,
        criterion=criterion,
        optimizer=optimizer,
        scaler=scaler,  # <--- Added the required scaler argument here
        device=device,
        epoch=1,
    )

    # 3. Get the new weights
    updated_weights = list(dummy_model.parameters())[0].data

    # Assertions
    assert isinstance(epoch_loss, float), "Loss should be a float"
    assert epoch_loss > 0, "Loss should be greater than zero"

    # CRITICAL: Prove the model actually learned something (weights changed)
    assert not torch.equal(
        initial_weights, updated_weights
    ), "Weights did NOT update during training!"

    # Prove the model was put in training mode
    assert dummy_model.training is True, "model.train() was not called"


def test_evaluate_does_not_update_weights_and_returns_probs(
    dummy_model, dummy_loader, criterion, device, num_classes
):
    """
    Test that evaluate computes loss, returns valid probabilities,
    and strictly DOES NOT change model weights.
    """
    # 1. Save a deep copy of the initial weights
    initial_weights = deepcopy(list(dummy_model.parameters())[0].data)

    # 2. Run evaluation
    val_loss, val_preds, val_targets = evaluate(
        model=dummy_model,
        dataloader=dummy_loader,
        criterion=criterion,
        device=device,
        phase="Val",
    )

    # 3. Get the weights after evaluation
    post_eval_weights = list(dummy_model.parameters())[0].data

    # Assertions for Loss
    assert isinstance(val_loss, float)
    assert val_loss > 0

    # CRITICAL: Prove the model did NOT learn (weights are identical)
    assert torch.equal(
        initial_weights, post_eval_weights
    ), "Weights changed during evaluation! (Leakage)"

    # Prove the model was put in evaluation mode
    assert dummy_model.training is False, "model.eval() was not called"

    # Assertions for Outputs
    num_samples = len(dummy_loader.dataset)
    assert isinstance(val_preds, torch.Tensor)
    assert isinstance(val_targets, torch.Tensor)
    assert val_preds.shape == (num_samples, num_classes)
    assert val_targets.shape == (num_samples, num_classes)

    # CRITICAL: Since evaluate uses torch.sigmoid(), all predictions MUST be between 0 and 1
    assert torch.all(val_preds >= 0.0) and torch.all(
        val_preds <= 1.0
    ), "Predictions are not valid probabilities [0, 1]"
