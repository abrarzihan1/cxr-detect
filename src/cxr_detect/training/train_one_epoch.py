import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.amp import autocast, GradScaler
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    max_grad_norm: float = 1.0,
) -> float:
    model.train()
    running_loss = 0.0
    total_samples = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", leave=False)

    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        # 1. Forward pass with Mixed Precision
        with autocast(device_type="cuda"):
            outputs = model(images)
            loss = criterion(outputs, targets)

        # 2. Backward pass scaled
        scaler.scale(loss).backward()

        # 3. Gradient Clipping (must unscale first!)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        # 4. Optimizer step
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return running_loss / total_samples
