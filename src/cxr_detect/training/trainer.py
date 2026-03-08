import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter

from cxr_detect.training.train_one_epoch import train_one_epoch
from cxr_detect.training.evaluate import evaluate
from cxr_detect.metrics.classification_metrics import compute_auroc
from cxr_detect.utils.config import load_disease_classes

LOGGER = logging.getLogger(__name__)


class Trainer:
    """
    Orchestrator for multi-epoch training.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
        save_dir: str | Path,
        num_epochs: int = 10,
        scheduler=None,
        patience: int = 5,
        max_grad_norm: float = 1.0,
        log_dir: Optional[str | Path] = None,
        experiment_name: str = "baseline",
        use_amp: bool = True,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.save_dir = Path(save_dir)
        self.num_epochs = num_epochs
        self.scheduler = scheduler
        self.max_grad_norm = max_grad_norm

        self.class_names = load_disease_classes()

        self.patience = patience
        self.early_stop_counter = 0
        self.best_val_loss = float("inf")
        self.start_epoch = 1

        self.save_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard setup
        self.writer = None
        if log_dir is not None:
            tb_dir = Path(log_dir) / experiment_name
            tb_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(tb_dir))

        # Enable AMP only if requested AND we are on a GPU
        scaler_enabled = bool(use_amp and self.device.type == "cuda")
        self.scaler = GradScaler("cuda", enabled=scaler_enabled)

    def fit(self):
        """Executes the multi-epoch training loop."""
        LOGGER.info(
            "Starting multi-epoch training on %s | AMP Enabled: %s | Patience: %d",
            self.device,
            self.scaler.is_enabled(),
            self.patience,
        )

        for epoch in range(self.start_epoch, self.num_epochs + 1):
            LOGGER.info("\n--- Epoch %d/%d ---", epoch, self.num_epochs)

            # 1. Train one epoch
            train_loss = train_one_epoch(
                model=self.model,
                dataloader=self.train_loader,
                criterion=self.criterion,
                optimizer=self.optimizer,
                scaler=self.scaler,
                device=self.device,
                epoch=epoch,
                max_grad_norm=self.max_grad_norm,
            )

            # 2. Evaluate
            val_loss, val_preds, val_targets = evaluate(
                model=self.model,
                dataloader=self.val_loader,
                criterion=self.criterion,
                device=self.device,
                phase="Val",
            )

            # 3. Compute Metrics
            macro_auc, class_aucs = compute_auroc(
                val_targets.numpy(), val_preds.numpy(), self.class_names
            )

            # Log to console
            auc_display = (
                f"{macro_auc:.4f}"
                if not np.isnan(macro_auc)
                else "N/A (Missing classes)"
            )
            LOGGER.info(
                "Train Loss: %.4f | Val Loss: %.4f | Val Macro AUROC: %s",
                train_loss,
                val_loss,
                auc_display,
            )

            # 4. Log to TensorBoard
            if self.writer is not None:
                self.writer.add_scalar("Loss/Train", train_loss, epoch)
                self.writer.add_scalar("Loss/Validation", val_loss, epoch)
                self.writer.add_scalar(
                    "Learning_Rate", self.optimizer.param_groups[0]["lr"], epoch
                )

                if not np.isnan(macro_auc):
                    self.writer.add_scalar("Metrics/Macro_AUROC", macro_auc, epoch)

                for disease, auc in class_aucs.items():
                    if not np.isnan(auc):
                        self.writer.add_scalar(f"Disease_AUC/{disease}", auc, epoch)

            # 5. Step Scheduler
            if self.scheduler is not None:
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # 6. Checkpointing & Early Stopping
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1

            self._save_checkpoint(epoch, val_loss, is_best)

            if self.early_stop_counter >= self.patience:
                LOGGER.info("Early stopping triggered after %d epochs.", epoch)
                break

        if self.writer is not None:
            self.writer.close()

    def _save_checkpoint(self, epoch: int, val_loss: float, is_best: bool):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler else None
            ),
            "best_val_loss": self.best_val_loss,
            "early_stop_counter": self.early_stop_counter,
        }

        torch.save(checkpoint, self.save_dir / "checkpoint_last.pth")
        if is_best:
            torch.save(checkpoint, self.save_dir / "checkpoint_best.pth")
            LOGGER.info("-> Saved new best model (Val Loss: %.4f)", val_loss)

    def load_checkpoint(self, path: str | Path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        LOGGER.info("Loading checkpoint from %s", path)
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.early_stop_counter = checkpoint.get("early_stop_counter", 0)
        self.start_epoch = checkpoint["epoch"] + 1
