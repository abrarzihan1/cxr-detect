import torch
import pandas as pd
from typing import List


def get_pos_weight(csv_path: str, disease_classes: List[str]) -> torch.Tensor:
    """
    Calculates positive class weights for BCEWithLogitsLoss.
    Formula: pos_weight = (Total Negative Samples) / (Total Positive Samples)
    """
    df = pd.read_csv(csv_path)

    pos_weights = []
    for disease in disease_classes:
        pos_count = df[disease].sum()
        neg_count = len(df) - pos_count

        # Prevent division by zero
        weight = neg_count / pos_count if pos_count > 0 else 1.0
        pos_weights.append(weight)

    return torch.tensor(pos_weights, dtype=torch.float32)
