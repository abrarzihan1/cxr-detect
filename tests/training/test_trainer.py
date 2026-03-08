from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

from cxr_detect.training.trainer import Trainer


def make_loader():
    x = torch.randn(4, 3)
    y = torch.randint(0, 2, (4, 1)).float()
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=2)


def make_trainer(tmp_path, scheduler=None, log_dir=None, patience=2, use_amp=True):
    model = nn.Linear(3, 1)
    optimizer = SGD(model.parameters(), lr=0.1)
    criterion = nn.BCEWithLogitsLoss()
    train_loader = make_loader()
    val_loader = make_loader()

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=torch.device("cpu"),
        save_dir=tmp_path / "ckpts",
        num_epochs=5,
        scheduler=scheduler,
        patience=patience,
        log_dir=log_dir,
        experiment_name="test_exp",
        use_amp=use_amp,
    )
    return trainer


@patch(
    "cxr_detect.training.trainer.load_disease_classes",
    return_value=["Atelectasis", "Cardiomegaly"],
)
def test_init_creates_save_dir_and_disables_amp_on_cpu(mock_classes, tmp_path):
    trainer = make_trainer(tmp_path, log_dir=None, use_amp=True)

    assert trainer.save_dir.exists()
    assert trainer.writer is None
    assert trainer.scaler.is_enabled() is False
    assert trainer.best_val_loss == float("inf")
    assert trainer.start_epoch == 1
    assert trainer.class_names == ["Atelectasis", "Cardiomegaly"]


@patch("cxr_detect.training.trainer.SummaryWriter")
@patch(
    "cxr_detect.training.trainer.load_disease_classes",
    return_value=["Atelectasis", "Cardiomegaly"],
)
def test_init_creates_tensorboard_writer(mock_classes, mock_writer_cls, tmp_path):
    trainer = make_trainer(tmp_path, log_dir=tmp_path / "runs")

    assert trainer.writer is mock_writer_cls.return_value
    mock_writer_cls.assert_called_once()
    called_log_dir = Path(mock_writer_cls.call_args.kwargs["log_dir"])
    assert called_log_dir == tmp_path / "runs" / "test_exp"


@patch("cxr_detect.training.trainer.compute_auroc")
@patch("cxr_detect.training.trainer.evaluate")
@patch("cxr_detect.training.trainer.train_one_epoch")
@patch(
    "cxr_detect.training.trainer.load_disease_classes",
    return_value=["Atelectasis", "Cardiomegaly"],
)
def test_fit_runs_one_epoch_and_logs_metrics(
    mock_classes,
    mock_train_one_epoch,
    mock_evaluate,
    mock_compute_auroc,
    tmp_path,
):
    trainer = make_trainer(tmp_path)
    trainer.num_epochs = 1
    trainer.writer = MagicMock()
    trainer._save_checkpoint = MagicMock()

    mock_train_one_epoch.return_value = 0.25
    mock_evaluate.return_value = (
        0.20,
        torch.tensor([[0.9, 0.1], [0.8, 0.2]]),
        torch.tensor([[1, 0], [1, 0]]),
    )
    mock_compute_auroc.return_value = (
        0.88,
        {"Atelectasis": 0.91, "Cardiomegaly": 0.85},
    )

    trainer.fit()

    mock_train_one_epoch.assert_called_once()
    mock_evaluate.assert_called_once()
    mock_compute_auroc.assert_called_once()
    trainer._save_checkpoint.assert_called_once_with(1, 0.20, True)

    trainer.writer.add_scalar.assert_any_call("Loss/Train", 0.25, 1)
    trainer.writer.add_scalar.assert_any_call("Loss/Validation", 0.20, 1)
    trainer.writer.add_scalar.assert_any_call("Learning_Rate", 0.1, 1)
    trainer.writer.add_scalar.assert_any_call("Metrics/Macro_AUROC", 0.88, 1)
    trainer.writer.add_scalar.assert_any_call("Disease_AUC/Atelectasis", 0.91, 1)
    trainer.writer.add_scalar.assert_any_call("Disease_AUC/Cardiomegaly", 0.85, 1)
    trainer.writer.close.assert_called_once()


@patch(
    "cxr_detect.training.trainer.compute_auroc",
    return_value=(float("nan"), {"Atelectasis": float("nan")}),
)
@patch("cxr_detect.training.trainer.evaluate")
@patch("cxr_detect.training.trainer.train_one_epoch", return_value=0.3)
@patch("cxr_detect.training.trainer.load_disease_classes", return_value=["Atelectasis"])
def test_fit_skips_nan_auc_tensorboard_logging(
    mock_classes,
    mock_train_one_epoch,
    mock_evaluate,
    mock_compute_auroc,
    tmp_path,
):
    trainer = make_trainer(tmp_path)
    trainer.num_epochs = 1
    trainer.writer = MagicMock()
    trainer._save_checkpoint = MagicMock()

    mock_evaluate.return_value = (
        0.4,
        torch.tensor([[0.7], [0.6]]),
        torch.tensor([[1], [1]]),
    )

    trainer.fit()

    add_scalar_calls = trainer.writer.add_scalar.call_args_list
    assert call("Metrics/Macro_AUROC", float("nan"), 1) not in add_scalar_calls
    assert call("Disease_AUC/Atelectasis", float("nan"), 1) not in add_scalar_calls


@patch(
    "cxr_detect.training.trainer.compute_auroc",
    return_value=(0.7, {"Atelectasis": 0.7}),
)
@patch("cxr_detect.training.trainer.evaluate")
@patch("cxr_detect.training.trainer.train_one_epoch", return_value=0.5)
@patch("cxr_detect.training.trainer.load_disease_classes", return_value=["Atelectasis"])
def test_early_stopping_triggers_after_patience(
    mock_classes,
    mock_train_one_epoch,
    mock_evaluate,
    mock_compute_auroc,
    tmp_path,
):
    trainer = make_trainer(tmp_path, patience=2)
    trainer.writer = None
    trainer._save_checkpoint = MagicMock()

    mock_evaluate.side_effect = [
        (0.5, torch.tensor([[0.1]]), torch.tensor([[0]])),
        (0.6, torch.tensor([[0.1]]), torch.tensor([[0]])),
        (0.7, torch.tensor([[0.1]]), torch.tensor([[0]])),
        (0.8, torch.tensor([[0.1]]), torch.tensor([[0]])),
    ]

    trainer.fit()

    assert mock_train_one_epoch.call_count == 3
    assert mock_evaluate.call_count == 3
    assert trainer.early_stop_counter == 2
    assert trainer.best_val_loss == 0.5


@patch("cxr_detect.training.trainer.load_disease_classes", return_value=["Atelectasis"])
def test_scheduler_step_called_for_regular_scheduler(mock_classes, tmp_path):
    scheduler = MagicMock()
    trainer = make_trainer(tmp_path, scheduler=scheduler)
    trainer.num_epochs = 1
    trainer.writer = None

    with patch("cxr_detect.training.trainer.train_one_epoch", return_value=0.1), patch(
        "cxr_detect.training.trainer.evaluate",
        return_value=(0.2, torch.tensor([[0.5]]), torch.tensor([[1]])),
    ), patch(
        "cxr_detect.training.trainer.compute_auroc",
        return_value=(0.9, {"Atelectasis": 0.9}),
    ), patch.object(trainer, "_save_checkpoint"):
        trainer.fit()

    scheduler.step.assert_called_once_with()


@patch("cxr_detect.training.trainer.load_disease_classes", return_value=["Atelectasis"])
def test_scheduler_step_called_with_val_loss_for_reduce_on_plateau(
    mock_classes, tmp_path
):
    model = nn.Linear(3, 1)
    optimizer = SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    trainer = Trainer(
        model=model,
        train_loader=make_loader(),
        val_loader=make_loader(),
        criterion=nn.BCEWithLogitsLoss(),
        optimizer=optimizer,
        device=torch.device("cpu"),
        save_dir=tmp_path / "ckpts",
        num_epochs=1,
        scheduler=scheduler,
        patience=2,
    )
    trainer.writer = None

    with patch("cxr_detect.training.trainer.train_one_epoch", return_value=0.1), patch(
        "cxr_detect.training.trainer.evaluate",
        return_value=(0.2, torch.tensor([[0.5]]), torch.tensor([[1]])),
    ), patch(
        "cxr_detect.training.trainer.compute_auroc",
        return_value=(0.9, {"Atelectasis": 0.9}),
    ), patch.object(trainer, "_save_checkpoint"), patch.object(
        scheduler, "step"
    ) as mock_step:
        trainer.fit()

    mock_step.assert_called_once_with(0.2)


@patch("cxr_detect.training.trainer.torch.save")
@patch("cxr_detect.training.trainer.load_disease_classes", return_value=["Atelectasis"])
def test_save_checkpoint_saves_last_and_best(mock_classes, mock_torch_save, tmp_path):
    trainer = make_trainer(tmp_path)
    trainer.best_val_loss = 0.3
    trainer.early_stop_counter = 1

    trainer._save_checkpoint(epoch=2, val_loss=0.3, is_best=True)

    assert mock_torch_save.call_count == 2
    saved_paths = [args[1] for args, _ in mock_torch_save.call_args_list]
    assert trainer.save_dir / "checkpoint_last.pth" in saved_paths
    assert trainer.save_dir / "checkpoint_best.pth" in saved_paths


@patch("cxr_detect.training.trainer.torch.load")
@patch("cxr_detect.training.trainer.load_disease_classes", return_value=["Atelectasis"])
def test_load_checkpoint_restores_state(mock_classes, mock_torch_load, tmp_path):
    scheduler = MagicMock()
    trainer = make_trainer(tmp_path, scheduler=scheduler)

    trainer.model.load_state_dict = MagicMock()
    trainer.optimizer.load_state_dict = MagicMock()
    trainer.scaler.load_state_dict = MagicMock()

    ckpt_path = tmp_path / "checkpoint.pth"
    ckpt_path.write_text("x")

    mock_torch_load.return_value = {
        "epoch": 4,
        "model_state_dict": {"w": 1},
        "optimizer_state_dict": {"lr": 0.1},
        "scaler_state_dict": {"scale": 1024},
        "scheduler_state_dict": {"step": 3},
        "best_val_loss": 0.123,
        "early_stop_counter": 2,
    }

    trainer.load_checkpoint(ckpt_path)

    trainer.model.load_state_dict.assert_called_once_with({"w": 1})
    trainer.optimizer.load_state_dict.assert_called_once_with({"lr": 0.1})
    trainer.scaler.load_state_dict.assert_called_once_with({"scale": 1024})
    scheduler.load_state_dict.assert_called_once_with({"step": 3})
    assert trainer.best_val_loss == 0.123
    assert trainer.early_stop_counter == 2
    assert trainer.start_epoch == 5


@patch("cxr_detect.training.trainer.load_disease_classes", return_value=["Atelectasis"])
def test_load_checkpoint_raises_for_missing_file(mock_classes, tmp_path):
    trainer = make_trainer(tmp_path)

    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        trainer.load_checkpoint(tmp_path / "missing.pth")
