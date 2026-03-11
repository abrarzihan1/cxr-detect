import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import AUROC
import numpy as np
import json
import csv

from cxr_detect.data.dataloader import create_dataloaders
from cxr_detect.models.model_factory import create_model
from cxr_detect.training.trainer import Trainer
from cxr_detect.utils.config import load_disease_classes
from cxr_detect.utils.loss import get_pos_weight
from cxr_detect.utils.env import set_seed

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
LOGGER = logging.getLogger(__name__)


def parse_args():
    """Only parse the path to the YAML config file."""
    parser = argparse.ArgumentParser(
        description="Multi-Epoch Training Script for CXR-Detect"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file (e.g., configs/baseline.yaml)",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Loads the YAML configuration file into a dictionary."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def compute_disease_auroc(model, test_loader, device, disease_classes):
    """Compute AUROC for each disease class on test set."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_images, batch_labels in test_loader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_images)
            probs = torch.sigmoid(outputs)
            all_preds.append(probs.cpu())
            all_targets.append(batch_labels.cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    # Compute per-disease AUROC
    aurocs = {}
    for i, disease in enumerate(disease_classes):
        target = all_targets[:, i]
        pred = all_preds[:, i]

        # Skip if no positive samples for this disease
        if target.sum() == 0 or (target == 0).sum() == 0:
            aurocs[disease] = 0.0
            LOGGER.warning(f"No positive/negative samples for {disease}, AUROC=0.0")
            continue

        metric = AUROC(task="binary")
        metric.update(pred, target.int())
        auroc_score = metric.compute().item()
        aurocs[disease] = auroc_score

    # Compute macro AUROC (average across diseases)
    valid_aurocs = [score for score in aurocs.values() if score > 0]
    macro_auroc = np.mean(valid_aurocs) if valid_aurocs else 0.0

    return aurocs, macro_auroc


def main():
    args = parse_args()
    config = load_config(args.config)

    # 1. Setup Environment
    seed = config.get("seed", 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiment_name = config.get("experiment_name", "baseline_run")
    LOGGER.info("Starting experiment: %s | Device: %s", experiment_name, device)

    # 2. Setup DataLoaders
    data_cfg = config["data"]
    train_loader, val_loader, test_loader = create_dataloaders(
        raw_csv_path=data_cfg["raw_csv_path"],
        processed_dir=data_cfg["processed_dir"],
        img_dir=data_cfg["img_dir"],
        batch_size=data_cfg.get("batch_size", 32),
        num_workers=data_cfg.get("num_workers", 4),
        img_size=data_cfg.get("img_size", 224),
        seed=seed,
    )

    disease_classes = load_disease_classes()

    # 3. Setup Model
    model_cfg = config["model"]
    model = create_model(
        model_name=model_cfg.get("name", "resnet50"),
        num_classes=len(disease_classes),
        pretrained=model_cfg.get("pretrained", True),
    )
    model = model.to(device)

    # 4. Setup Loss (with class imbalance weights calculated from the train set)
    train_csv = Path(data_cfg["processed_dir"]) / "train.csv"
    if train_csv.exists():
        pos_weight = get_pos_weight(str(train_csv), disease_classes).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        LOGGER.warning("Processed train.csv not found for weights. Using standard BCE.")
        criterion = nn.BCEWithLogitsLoss()

    # 5. Setup Optimizer & Scheduler
    train_cfg = config["training"]
    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg.get("learning_rate", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )

    epochs = train_cfg.get("epochs", 20)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # 6. Initialize Multi-Epoch Trainer & Experiment Folders
    exp_save_dir = Path(train_cfg.get("save_dir", "experiments")) / experiment_name
    exp_save_dir.mkdir(parents=True, exist_ok=True)

    log_dir = train_cfg.get("log_dir", "logs/tensorboard")

    # Save the exact config used for this run
    config_out_path = exp_save_dir / "config.yaml"
    with open(config_out_path, "w") as f:
        yaml.safe_dump(config, f)
    LOGGER.info("Saved config to %s", config_out_path)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        save_dir=exp_save_dir,
        num_epochs=epochs,
        scheduler=scheduler,
        patience=train_cfg.get("patience", 5),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        log_dir=log_dir,
        experiment_name=experiment_name,
        use_amp=train_cfg.get("use_amp", True),
    )

    # Resume training if a checkpoint is provided in config
    resume_path = train_cfg.get("resume", None)
    if resume_path:
        trainer.load_checkpoint(resume_path)

    # 7. Start the Training Loop
    history = trainer.fit()
    LOGGER.info("Experiment %s finished.", experiment_name)

    # Save training log to CSV
    log_csv_path = exp_save_dir / "train_log.csv"
    with open(log_csv_path, "w", newline="") as csvfile:
        fieldnames = ["epoch", "train_loss", "val_loss", "val_auc", "learning_rate"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(
                {
                    "epoch": row.get("epoch"),
                    "train_loss": row.get("train_loss"),
                    "val_loss": row.get("val_loss"),
                    "val_auc": row.get("val_auc"),
                    "learning_rate": row.get("learning_rate"),
                }
            )
    LOGGER.info("Saved train log to %s", log_csv_path)

    # 8. Compute and log disease-wise AUROC on test set (using best model)
    LOGGER.info("Computing disease-wise AUROC on test set...")
    best_model_path = exp_save_dir / "best_model.pt"

    if best_model_path.exists():
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        LOGGER.info("Loaded best model for evaluation")

        disease_aurocs, macro_auroc = compute_disease_auroc(
            model, test_loader, device, disease_classes
        )

        # Log results
        LOGGER.info("=== Disease-wise AUROC Results ===")
        for disease, auroc in disease_aurocs.items():
            LOGGER.info("%-25s: %.4f", disease, auroc)
        LOGGER.info("Macro AUROC (avg): %.4f", macro_auroc)

        # Save Final Metrics as JSON
        mean_auc = (
            float(np.mean(list(disease_aurocs.values()))) if disease_aurocs else 0.0
        )
        micro_auc = (
            mean_auc  # Placeholder: adjust if you calculate true micro_auc separately
        )

        metrics = {
            "mean_auc": mean_auc,
            "macro_auc": macro_auroc,
            "micro_auc": micro_auc,
            "per_class_auc": disease_aurocs,
        }

        metrics_path = exp_save_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        LOGGER.info("Saved metrics to %s", metrics_path)
    else:
        LOGGER.warning(
            "Best model not found at %s. Skipping metrics generation.", best_model_path
        )


if __name__ == "__main__":
    main()
