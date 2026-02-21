from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image

from cxr_detect.utils.config import load_disease_classes
from cxr_detect.utils.process_data import load_data


class CXRDataset(Dataset):
    def __init__(self, csv_path: str | Path, img_dir: str | Path, transform=None):
        """
        Args:
            csv_path: Path to the CSV file containing image IDs and labels.
            img_dir: Path to the directory containing images.
            transform: Optional transform to be applied on a sample.
        """
        self.csv_path = Path(csv_path)
        self.img_dir = Path(img_dir)
        self.transform = transform

        # 1. Load the dataframe
        self.df = load_data(csv_path)

        # 2. Load disease classes
        # This ensures the model only trains on the classes defined in your config
        self.disease_classes = load_disease_classes()

        # Validate that config classes actually exist in the CSV
        missing_cols = [c for c in self.disease_classes if c not in self.df.columns]
        if missing_cols:
            raise ValueError(
                f"The following config classes are missing from CSV: {missing_cols}"
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Returns:
            image: Transformed image tensor
            label: Binary vector of shape (14,) for multi-label classification
            (optionally) metadata: dict with additional info
        """
        # Get image filename
        img_name = self.df.iloc[idx]["image_id"]
        img_path = self.img_dir / img_name

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)

        # Extract multi-label vector (14 binary labels)
        label = self.df.iloc[idx][self.disease_classes].values.astype("float32")
        label = torch.tensor(label, dtype=torch.float32)

        return image, label
