import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast
from typing import Tuple
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    phase: str = "Val",
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    model.eval()
    running_loss = 0.0
    total_samples = 0

    all_preds = []
    all_targets = []

    pbar = tqdm(dataloader, desc=f"[{phase}]", leave=False)

    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Mixed precision inference
        with autocast(device_type="cuda"):
            logits = model(images)
            loss = criterion(logits, targets)

        probs = torch.sigmoid(logits)

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

        all_preds.append(probs.cpu())
        all_targets.append(targets.cpu())

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    all_preds_tensor = torch.cat(all_preds, dim=0)
    all_targets_tensor = torch.cat(all_targets, dim=0)

    return running_loss / total_samples, all_preds_tensor, all_targets_tensor
