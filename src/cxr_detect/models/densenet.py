import torch
import torch.nn as nn
from torchvision import models


class CXRDenseNet(nn.Module):
    """
    Chest X-Ray DenseNet Model.
    Uses an ImageNet pre-trained backbone and replaces the final fully
    connected layer to output `num_classes` logits.
    """

    def __init__(
        self, num_classes: int, model_name: str = "densenet121", pretrained: bool = True
    ):
        super().__init__()

        # Use the modern 'DEFAULT' weights API in PyTorch
        weights = "DEFAULT" if pretrained else None

        if model_name == "densenet121":
            self.backbone = models.densenet121(weights=weights)
        elif model_name == "densenet169":
            self.backbone = models.densenet169(weights=weights)
        elif model_name == "densenet201":
            self.backbone = models.densenet201(weights=weights)
        else:
            raise ValueError(f"Unsupported DenseNet architecture: {model_name}")

        # Extract the number of input features to the final classifier layer
        in_features = self.backbone.classifier.in_features

        # Replace the final layer
        # PyTorch's BCEWithLogitsLoss requires raw logits, so no Sigmoid is added here
        self.backbone.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        Args:
            x: Input tensor of shape (Batch, 3, Height, Width)
        Returns:
            Logits tensor of shape (Batch, num_classes)
        """
        return self.backbone(x)
