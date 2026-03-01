import numpy as np
from sklearn.metrics import roc_auc_score
from typing import Dict, List, Tuple


def compute_auroc(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]
) -> Tuple[float, Dict[str, float]]:
    """
    Computes Macro AUROC and Per-Class AUROC.
    Safely handles missing classes (e.g., in small validation batches).
    """
    class_aucs = {}

    for i, class_name in enumerate(class_names):
        try:
            # Calculate AUC for this specific disease
            auc = roc_auc_score(y_true[:, i], y_pred[:, i])
            class_aucs[class_name] = auc
        except ValueError:
            # Happens if a class has no positive examples in the validation batch
            class_aucs[class_name] = float("nan")

    # Calculate Macro AUC, ignoring NaNs
    valid_aucs = [auc for auc in class_aucs.values() if not np.isnan(auc)]
    macro_auc = np.mean(valid_aucs) if valid_aucs else float("nan")

    return macro_auc, class_aucs
