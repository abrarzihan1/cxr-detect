from torchvision import transforms

# Standard ImageNet statistics
# Required because we are using pre-trained models (which expect these values)
# and we convert our grayscale X-rays to RGB in the Dataset class.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(img_size: int = 224):
    """
    Returns transforms for the training set, including data augmentation.

    Augmentations selected for Chest X-Rays:
    RandomResizedCrop: simulating different zoom levels/patient distances.
    """
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_val_transforms(img_size: int = 224):
    """
    Returns deterministic transforms for validation.
    No augmentation, just resizing and normalization.
    """
    return transforms.Compose(
        [
            # If images are already 224x224, this acts as a safeguard.
            # If larger, it resizes them to the input requirement.
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_test_transforms(img_size: int = 224):
    """
    Returns transforms for testing. Usually identical to validation.
    """
    return get_val_transforms(img_size)
