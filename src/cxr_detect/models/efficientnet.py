import torch
import torch.nn as nn
from torchvision import models


class CXREfficientNet(nn.Module):
    """
    Chest X-Ray EfficientNet Model.
    Uses an ImageNet pre-trained backbone and replaces the final fully
    connected layer to output `num_classes` logits.
    """

    def __init__(
        self,
        num_classes: int,
        model_name: str = "efficientnet_b0",
        pretrained: bool = True,
    ):
        super().__init__()

        # Use the modern 'DEFAULT' weights API in PyTorch
        weights = "DEFAULT" if pretrained else None

        if model_name == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(weights=weights)
        elif model_name == "efficientnet_b4":
            self.backbone = models.efficientnet_b4(weights=weights)
        elif model_name == "efficientnet_b7":
            self.backbone = models.efficientnet_b7(weights=weights)
        else:
            raise ValueError(f"Unsupported EfficientNet architecture: {model_name}")

        # EfficientNet's classifier is a Sequential block: (0): Dropout, (1): Linear
        # Extract the number of input features from the Linear layer
        in_features = self.backbone.classifier[1].in_features

        # Replace the final linear layer
        # PyTorch's BCEWithLogitsLoss requires raw logits, so no Sigmoid is added
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        Args:
            x: Input tensor of shape (Batch, 3, Height, Width)
        Returns:
            Logits tensor of shape (Batch, num_classes)
        """
        return self.backbone(x)
