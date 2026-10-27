import json
from pathlib import Path
from typing import Union
from backend.app.evaluation.schemas import BenchmarkDataset

def load_dataset(path_or_dict: Union[str, Path, dict]) -> BenchmarkDataset:
    if isinstance(path_or_dict, dict):
        data = path_or_dict
    else:
        path = Path(path_or_dict)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

    return BenchmarkDataset(**data)
