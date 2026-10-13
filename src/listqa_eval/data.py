"""
Data loading utilities for ListQA evaluation.

Supports loading datasets from:
- Local JSONL files
- HuggingFace Hub datasets
- Raw Python dicts / lists

Each sample must have at minimum: ``"problem"`` (question) and ``"answer"``
(reference answer). Optionally ``"metadata"`` (JSON string).
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def load_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dicts."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning("Skipping invalid JSON on line %d: %s", line_num, e)
    logger.info("Loaded %d samples from %s", len(data), path)
    return data


def load_from_huggingface(
    dataset_name: str,
    split: str = "test",
    subset: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Load a dataset from the HuggingFace Hub.

    Args:
        dataset_name: HuggingFace dataset identifier (e.g. ``"AmazonScience/ListQA"``).
        split: Dataset split to load (default ``"test"``).
        subset: Optional dataset subset/configuration name.

    Returns:
        List of sample dicts.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' library is required to load from HuggingFace Hub. "
            "Install it with: pip install datasets"
        )

    kwargs: Dict[str, Any] = {"path": dataset_name, "split": split}
    if subset:
        kwargs["name"] = subset

    logger.info("Loading dataset '%s' (split=%s, subset=%s) from HuggingFace Hub",
                dataset_name, split, subset)
    ds = load_dataset(**kwargs)
    data = [dict(row) for row in ds]
    logger.info("Loaded %d samples from HuggingFace dataset '%s'", len(data), dataset_name)
    return data


def load_predictions_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load model predictions from a JSONL file.

    Each line should be a JSON object with at least ``"problem"`` and
    ``"prediction"`` keys (the model's generated answer).
    """
    return load_jsonl(path)


def validate_dataset(data: List[Dict[str, Any]], require_predictions: bool = False) -> None:
    """
    Validate that a dataset has the required fields.

    Raises ValueError if required fields are missing.
    """
    required = {"problem", "answer"}
    if require_predictions:
        required.add("prediction")

    for i, sample in enumerate(data):
        missing = required - set(sample.keys())
        if missing:
            raise ValueError(
                f"Sample {i} is missing required fields: {missing}. "
                f"Available keys: {list(sample.keys())}"
            )
