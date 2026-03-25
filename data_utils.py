

from datasets import load_dataset, Dataset
import config


REQUIRED_COLUMNS = {"prompt", "chosen", "rejected"}


def load_preference_dataset(csv_path: str = config.PREFERENCE_DATA_CSV) -> Dataset:
    """
    Load the pharma preference CSV and return the training split as a
    HuggingFace Dataset.

    Raises:
        ValueError - if any required column is missing.
        FileNotFoundError - if the CSV path doesn't exist.
    """
    raw = load_dataset("csv", data_files=csv_path)
    split = raw["train"]
    _validate_columns(split)
    print(f"[data] Loaded {len(split)} preference pairs from '{csv_path}'")
    return split


def _validate_columns(dataset: Dataset) -> None:
    """Assert that all DPO-required columns are present."""
    missing = REQUIRED_COLUMNS - set(dataset.column_names)
    if missing:
        raise ValueError(
            f"Preference dataset is missing required columns: {missing}. "
            f"Found columns: {dataset.column_names}"
        )
