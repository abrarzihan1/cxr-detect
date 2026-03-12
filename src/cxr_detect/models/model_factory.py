import logging
import torch.nn as nn

from cxr_detect.models.densenet import CXRDenseNet
from cxr_detect.models.resnet import CXRResNet
from cxr_detect.models.efficientnet import CXREfficientNet

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

    elif model_name.startswith("densenet"):
        LOGGER.info(
            f"Initializing {model_name} (classes={num_classes}, pretrained={pretrained})"
        )
        return CXRDenseNet(
            num_classes=num_classes, model_name=model_name, pretrained=pretrained
        )

    elif model_name.startswith("efficientnet"):
        LOGGER.info(
            f"Initializing {model_name} (classes={num_classes}, pretrained={pretrained})"
        )
        return CXREfficientNet(
            num_classes=num_classes, model_name=model_name, pretrained=pretrained
        )

    else:
        raise ValueError(
            f"Model architecture '{model_name}' is not supported. "
            f"Available options: resnet18, resnet34, resnet50, densenet121, densenet169, densenet201, "
            f"efficientnet_b0, efficientnet_b4, efficientnet_b7."
        )
