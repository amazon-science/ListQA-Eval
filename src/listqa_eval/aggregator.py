"""
Score aggregation for ListQA evaluation.

Computes overall and per-metadata-slice precision, recall, and F1 from
LLM-as-a-Judge scores.  Mirrors the logic from the original ListQAScoreAggregator
but without any framework dependencies.
"""

import json
from collections import defaultdict
from typing import Any, Dict, List, Optional


LLM_JUDGE_MAP = {"A": 1, "B": 0}


def _prf(num_samples: int, precision_scores: List[float], recall_scores: List[float]) -> Dict[str, Any]:
    """Compute precision, recall, F1 from accumulated per-sample scores."""
    if num_samples == 0:
        return {"num_samples": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    p = sum(precision_scores) / num_samples
    r = sum(recall_scores) / num_samples
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"num_samples": num_samples, "precision": p, "recall": r, "f1": f1}


def aggregate_llm_judge_scores(
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Aggregate per-sample LLM judge results into overall and sliced scores.

    Args:
        results: List of per-sample dicts, each containing:
            - ``"question"`` (str)
            - ``"answer"`` (str)
            - ``"prediction"`` (str)
            - ``"metadata"`` (str, optional) – JSON string with optional keys
              ``"country"``, ``"category"``, ``"year_range"``
            - ``"llm_judge"`` dict with:
                - ``"list_element_scores"`` – list of dicts with ``"score"``
                - ``"num_pred_list_elements"`` (int)
                - ``"num_target_list_elements"`` (int)

    Returns:
        Dict with ``"precision"``, ``"recall"``, ``"f1"``, ``"num_samples"``,
        and optional ``"country"``, ``"category"``, ``"year_range"`` breakdowns.
    """
    if not results:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "num_samples": 0}

    overall_precision: List[float] = []
    overall_recall: List[float] = []
    category_scores: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    year_scores: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    country_scores: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for result in results:
        judge_data = result["llm_judge"]
        num_pred = judge_data["num_pred_list_elements"]
        num_target = judge_data["num_target_list_elements"]
        element_scores = judge_data["list_element_scores"]

        correct = sum(LLM_JUDGE_MAP.get(es["score"], 0) for es in element_scores)
        precision = correct / num_pred if num_pred > 0 else 0.0
        recall = correct / num_target if num_target > 0 else 0.0
        overall_precision.append(precision)
        overall_recall.append(recall)

        # Metadata slicing
        metadata_str = result.get("metadata")
        if metadata_str:
            try:
                metadata = json.loads(metadata_str) if isinstance(metadata_str, str) else metadata_str
            except (json.JSONDecodeError, TypeError):
                metadata = {}

            if "country" in metadata:
                country_scores[metadata["country"]]["precision"].append(precision)
                country_scores[metadata["country"]]["recall"].append(recall)
            if "category" in metadata:
                category_scores[metadata["category"]]["precision"].append(precision)
                category_scores[metadata["category"]]["recall"].append(recall)
            if "year_range" in metadata:
                year_scores[metadata["year_range"]]["precision"].append(precision)
                year_scores[metadata["year_range"]]["recall"].append(recall)

    overall = _prf(len(results), overall_precision, overall_recall)

    # Add breakdowns
    if country_scores:
        overall["country"] = {
            c: _prf(len(s["precision"]), s["precision"], s["recall"])
            for c, s in country_scores.items()
        }
    if category_scores:
        overall["category"] = {
            c: _prf(len(s["precision"]), s["precision"], s["recall"])
            for c, s in category_scores.items()
        }
    if year_scores:
        overall["year_range"] = {
            y: _prf(len(s["precision"]), s["precision"], s["recall"])
            for y, s in year_scores.items()
        }

    return overall
