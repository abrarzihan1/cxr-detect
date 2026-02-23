import torch
import torch.nn as nn
from torchvision import models


class CXRResNet(nn.Module):
    """
    Chest X-Ray ResNet Model.
    Uses an ImageNet pre-trained backbone and replaces the final fully
    connected layer to output `num_classes` logits.
    """

    def __init__(
        self, num_classes: int, model_name: str = "resnet50", pretrained: bool = True
    ):
        super().__init__()

        # Use the modern 'DEFAULT' weights API in PyTorch (equivalent to IMAGENET1K_V1/V2)
        weights = "DEFAULT" if pretrained else None

        if model_name == "resnet18":
            self.backbone = models.resnet18(weights=weights)
        elif model_name == "resnet34":
            self.backbone = models.resnet34(weights=weights)
        elif model_name == "resnet50":
            self.backbone = models.resnet50(weights=weights)
        else:
            raise ValueError(f"Unsupported ResNet architecture: {model_name}")

        # Extract the number of input features to the final fully connected layer
        in_features = self.backbone.fc.in_features

        # Replace the final layer.
        # We do NOT add a Sigmoid layer here. PyTorch's BCEWithLogitsLoss
        # is numerically more stable when applied directly to raw logits.
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        Args:
            x: Input tensor of shape (Batch, 3, Height, Width)
        Returns:
            Logits tensor of shape (Batch, num_classes)
        """
        return self.backbone(x)
