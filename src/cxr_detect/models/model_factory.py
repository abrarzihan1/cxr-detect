import logging
import torch.nn as nn
from cxr_detect.models.resnet import CXRResNet

LOGGER = logging.getLogger(__name__)


def create_model(
    model_name: str, num_classes: int, pretrained: bool = True
) -> nn.Module:
    """
    Factory function to instantiate models based on a string name.

    Args:
        model_name: Name of the architecture (e.g., 'resnet18', 'resnet50').
        num_classes: Number of output disease classes.
        pretrained: Whether to initialize with ImageNet weights.

    Returns:
        An instantiated PyTorch nn.Module.
    """
    model_name = model_name.lower()

    if model_name.startswith("resnet"):
        LOGGER.info(
            f"Initializing {model_name} (classes={num_classes}, pretrained={pretrained})"
        )
        return CXRResNet(
            num_classes=num_classes, model_name=model_name, pretrained=pretrained
        )

    # Easily extensible in the future:
    # elif model_name.startswith("densenet"):
    #     return CXRDenseNet(...)

    else:
        raise ValueError(
            f"Model architecture '{model_name}' is not supported. "
            f"Available options: resnet18, resnet34, resnet50."
        )
