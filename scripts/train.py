import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

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


def main():
    args = parse_args()
    config = load_config(args.config)

    # 1. Setup Environment
    # We use .get() for optional parameters to prevent crashes if they are missing
    seed = config.get("seed", 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiment_name = config.get("experiment_name", "baseline_run")
    LOGGER.info("Starting experiment: %s | Device: %s", experiment_name, device)

    # 2. Setup DataLoaders
    # Notice we extract from the "data" section of the YAML
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

    # 6. Initialize Multi-Epoch Trainer
    exp_save_dir = (
        Path(train_cfg.get("save_dir", "models/checkpoints")) / experiment_name
    )
    log_dir = train_cfg.get("log_dir", "logs/tensorboard")

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
    trainer.fit()
    LOGGER.info("Experiment %s finished.", experiment_name)


if __name__ == "__main__":
    main()
