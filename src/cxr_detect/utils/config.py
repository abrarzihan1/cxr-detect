import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = PROJECT_ROOT / "configs"


def load_disease_classes():
    config_path = CONFIG_DIR / "disease_classes.json"

    with open(config_path, "r") as f:
        data = json.load(f)

    return data["classes"]
