"""
ListQA Evaluation - A standalone benchmark for evaluating list-formatted question answering.

This package provides metrics for evaluating list-formatted QA outputs including:
- Relaxed exact match (precision/recall) for list elements
- ROUGE-based list element matching with Hungarian assignment
- LLM-as-a-Judge scoring using locally-loaded HuggingFace models
- List format compliance checking
- Aggregated precision/recall/F1 with metadata breakdowns
"""

__version__ = "1.0.0"

from listqa_eval.metrics import (
    compute_list_format_em,
    compute_relaxed_precision_list_em,
    compute_relaxed_recall_list_em,
    compute_rouge_list_match,
    get_rouge_best_match_ranked_lists,
    normalize_answers,
    normalize_strings,
    parse_reference_list_elements,
)

__all__ = [
    "compute_relaxed_recall_list_em",
    "compute_relaxed_precision_list_em",
    "compute_list_format_em",
    "compute_rouge_list_match",
    "get_rouge_best_match_ranked_lists",
    "normalize_strings",
    "normalize_answers",
    "parse_reference_list_elements",
]
