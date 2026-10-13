"""
Standalone metrics for ListQA evaluation.

Provides list-level precision, recall, and F1 computations using:
- Relaxed exact-match (substring containment)
- ROUGE-based element matching with Hungarian assignment
- List format compliance checking

No external framework dependencies — only standard scientific Python libraries.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from evaluate import load as load_metric
from scipy.optimize import linear_sum_assignment

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MULTI_FACET_KEY = "multi-faceted"
SINGLE_FACET_KEY = "single-faceted"
ROUGE_FILTER_THRESHOLD = 0.0
ROUGE_CORRECT_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Text normalisation helpers
# ---------------------------------------------------------------------------


def normalize_strings(text: str) -> str:
    """Strip trailing spaces before newlines and lower-case the text."""
    if not isinstance(text, str):
        raise ValueError("Input must be a string.")
    return text.replace("\n ", "\n").lower()


def normalize_answers(text: str) -> str:
    """Remove extra generation of questions that may appear after Example 4."""
    if not isinstance(text, str):
        raise ValueError("Input must be a string.")
    return "\n" + text.split("\nExample 4:")[0].lower()


# ---------------------------------------------------------------------------
# List-element parsing
# ---------------------------------------------------------------------------


def parse_reference_list_elements(target: str) -> List[str]:
    """Extract numbered list elements from a target / prediction string."""
    list_pattern = r"\n\d+\.\s(.+?)(?=\n\d+\.|\Z)"
    list_elements = re.findall(list_pattern, target, re.DOTALL)
    return [elem.strip().lower() for elem in list_elements]


def parse_multi_faceted_list_element(element: str) -> Dict[str, Any]:
    """Parse a multi-faceted list element into main element + sub-elements."""
    pattern = r"^\* (.*?): (.*?)$"
    matches = re.finditer(pattern, element, re.MULTILINE)
    main_element = element.split("*")[0].strip()
    sub_elements = {m.group(1): m.group(2).strip() for m in matches}
    return {"main_element": main_element, "sub_elements": sub_elements}


def flatten_list(list_elements: List[str]) -> List[str]:
    """Flatten multi-faceted list elements into a flat list of strings."""
    flattened: List[str] = []
    for elem in list_elements:
        facet = parse_multi_faceted_list_element(elem)
        flattened.append(facet["main_element"])
        for sub_val in facet["sub_elements"].values():
            flattened.append(sub_val)
    return flattened


# ---------------------------------------------------------------------------
# Relaxed exact-match metrics
# ---------------------------------------------------------------------------


def compute_relaxed_recall_list_em(target: str, pred: str) -> float:
    """
    Relaxed recall: fraction of reference list elements found (as substrings)
    in the prediction.
    """
    target = normalize_strings(target).lower()
    pred = normalize_answers(pred).lower()

    list_em = 0
    target_list_elements = parse_reference_list_elements(target)
    num_list_elements = len(target_list_elements)
    if num_list_elements == 0:
        return 0.0

    for elem in target_list_elements:
        if "\n* " not in elem:
            if elem in pred:
                list_em += 1
        else:
            mf = parse_multi_faceted_list_element(elem)
            if mf["main_element"] in pred:
                list_em += 1
            for cat_val in mf["sub_elements"].values():
                if cat_val in pred:
                    list_em += 1
            num_list_elements += len(mf["sub_elements"])

    return list_em / num_list_elements


def compute_relaxed_precision_list_em(target: str, pred: str) -> float:
    """
    Relaxed precision: fraction of predicted list elements found (as substrings)
    in the reference.
    """
    target = normalize_strings(target).lower()
    pred = normalize_answers(pred).lower()

    pred_list_elements = parse_reference_list_elements(pred)
    target_list_elements = parse_reference_list_elements(target)
    flattened_targets = flatten_list(target_list_elements)
    num_list_elements = len(pred_list_elements)
    if num_list_elements == 0:
        return 0.0

    list_em = 0
    for elem in pred_list_elements:
        if "\n* " not in elem:
            if elem in target:
                list_em += 1
            else:
                list_em += _set_intersection_match(elem, flattened_targets)
        else:
            mf = parse_multi_faceted_list_element(elem)
            if mf["main_element"] in target:
                list_em += 1
            else:
                list_em += _set_intersection_match(mf["main_element"], flattened_targets)
            for cat_val in mf["sub_elements"].values():
                if cat_val in target:
                    list_em += 1
                else:
                    list_em += _set_intersection_match(cat_val, flattened_targets)
            num_list_elements += len(mf["sub_elements"])

    return list_em / num_list_elements


def _set_intersection_match(phrase: str, ref_list_elements: List[str]) -> int:
    for ref_ele in ref_list_elements:
        if ref_ele in phrase:
            return 1
    return 0


# ---------------------------------------------------------------------------
# List format compliance
# ---------------------------------------------------------------------------


def compute_list_format_em(pred: str, metadata_json: str) -> float:
    """
    Check whether the prediction follows the correct list format
    (single-faceted vs multi-faceted) as specified in the metadata.
    """
    prompt_type = json.loads(metadata_json)["prompt_type"]
    pred = normalize_answers(normalize_strings(pred))
    pred_list_elements = parse_reference_list_elements(pred)

    list_format_score = 0
    if len(pred_list_elements) < 3:
        return 0.0

    for elem in pred_list_elements:
        mf = parse_multi_faceted_list_element(elem)
        if prompt_type == SINGLE_FACET_KEY:
            if len(mf["sub_elements"]) == 0:
                list_format_score += 1
        elif prompt_type == MULTI_FACET_KEY:
            if len(mf["sub_elements"]) > 0:
                list_format_score += 1
        else:
            raise NotImplementedError(
                "There should not be any prompts beyond single and multi-faceted!"
            )

    return list_format_score / len(pred_list_elements)


# ---------------------------------------------------------------------------
# ROUGE-based list-element matching (Hungarian assignment)
# ---------------------------------------------------------------------------


def get_rouge_best_match_ranked_lists(
    target: str,
    pred: str,
    rouge_filter_threshold: float = ROUGE_FILTER_THRESHOLD,
    rouge_score_type: str = "rouge1",
) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    Use ROUGE + Hungarian assignment to find the best one-to-one matching
    between predicted and reference list elements.

    Returns:
        (list_pairs_to_judge, num_pred_elements, num_target_elements)
    """
    target = normalize_strings(target).lower()
    pred = normalize_answers(pred).lower()

    pred_list_elements = parse_reference_list_elements(pred)
    target_list_elements = parse_reference_list_elements(target)
    num_pred = len(pred_list_elements)
    num_target = len(target_list_elements)

    if num_pred == 0:
        return [{"pred": pred, "target": target, "rouge_score": 0.0}], 0, num_target

    rouge_metric = load_metric("rouge")
    scores_matrix: List[List[float]] = []

    for p_elem in pred_list_elements:
        row: List[float] = []
        for t_elem in target_list_elements:
            flat_pred = " ".join(flatten_list([p_elem]))
            flat_target = " ".join(flatten_list([t_elem]))
            rouge_scores = rouge_metric.compute(
                predictions=[flat_pred],
                references=[flat_target],
                use_stemmer=True,
            )
            row.append(rouge_scores[rouge_score_type])
        scores_matrix.append(row)

    # Hungarian assignment (minimise negative = maximise)
    cost_matrix = [[-s for s in row] for row in scores_matrix]
    pred_idx_list, tgt_idx_list = linear_sum_assignment(cost_matrix)
    assignment = {int(p): int(t) for p, t in zip(pred_idx_list, tgt_idx_list)}

    # Filter by threshold
    filtered: List[Tuple[Optional[int], Optional[float]]] = [(None, None)] * num_pred
    for p_idx, t_idx in assignment.items():
        if scores_matrix[p_idx][t_idx] >= rouge_filter_threshold:
            filtered[p_idx] = (t_idx, scores_matrix[p_idx][t_idx])

    # Build pairs
    list_pairs: List[Dict[str, Any]] = []
    for p_idx, p_elem in enumerate(pred_list_elements):
        t_idx, rouge_score = filtered[p_idx]
        t_elem = target_list_elements[t_idx] if t_idx is not None else None
        list_pairs.append({"pred": p_elem, "target": t_elem, "rouge_score": rouge_score})

    return list_pairs, num_pred, num_target


def compute_rouge_list_match(
    target: str,
    pred: str,
    rouge_filter_threshold: float = ROUGE_FILTER_THRESHOLD,
    rouge_correct_threshold: float = ROUGE_CORRECT_THRESHOLD,
    rouge_score_type: str = "rouge1",
) -> Dict[str, float]:
    """
    End-to-end ROUGE list-element matching returning precision, recall, and F1.
    """
    list_pairs, num_pred, num_target = get_rouge_best_match_ranked_lists(
        target, pred, rouge_filter_threshold, rouge_score_type
    )

    matched = 0
    for pair in list_pairs:
        if pair["rouge_score"] is not None and pair["rouge_score"] >= rouge_correct_threshold:
            matched += 1

    precision = matched / num_pred if num_pred > 0 else 0.0
    recall = matched / num_target if num_target > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "rouge_list_precision": precision,
        "rouge_list_recall": recall,
        "rouge_list_f1": f1,
        "num_pred_list_elements": num_pred,
        "num_target_list_elements": num_target,
        "list_pairs": list_pairs,
    }
