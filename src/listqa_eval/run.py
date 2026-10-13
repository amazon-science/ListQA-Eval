"""
Main CLI entry-point for running the ListQA evaluation pipeline.

Usage:
    listqa-eval --predictions predictions.jsonl --judge-model meta-llama/Llama-3.1-8B-Instruct
    listqa-eval --predictions predictions.jsonl --no-llm-judge   # ROUGE-only metrics
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

from listqa_eval.aggregator import aggregate_llm_judge_scores
from listqa_eval.data import load_jsonl, load_from_huggingface, validate_dataset
from listqa_eval.metrics import (
    compute_list_format_em,
    compute_relaxed_precision_list_em,
    compute_relaxed_recall_list_em,
    get_rouge_best_match_ranked_lists,
    compute_rouge_list_match,
)

logger = logging.getLogger("listqa_eval")


def evaluate_sample_rouge_only(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Compute all non-LLM metrics for a single sample."""
    question = sample["problem"]
    answer = sample["answer"]
    prediction = sample["prediction"]
    metadata = sample.get("metadata")

    result: Dict[str, Any] = {
        "question": question,
        "answer": answer,
        "prediction": prediction,
        "metadata": metadata,
    }

    # Relaxed EM
    result["recall_relaxed_em"] = compute_relaxed_recall_list_em(answer, prediction)
    result["precision_relaxed_em"] = compute_relaxed_precision_list_em(answer, prediction)

    # List format
    if metadata:
        try:
            result["list_format_em"] = compute_list_format_em(prediction, metadata)
        except Exception:
            result["list_format_em"] = None

    # ROUGE list match
    rouge_result = compute_rouge_list_match(answer, prediction)
    result["rouge_list_precision"] = rouge_result["rouge_list_precision"]
    result["rouge_list_recall"] = rouge_result["rouge_list_recall"]
    result["rouge_list_f1"] = rouge_result["rouge_list_f1"]

    return result


def evaluate_with_llm_judge(
    samples: List[Dict[str, Any]],
    judge_model_name: str,
    device: str = "auto",
    torch_dtype: str = "auto",
    max_retries: int = 3,
    trust_remote_code: bool = False,
) -> List[Dict[str, Any]]:
    """Run full evaluation including LLM-as-a-Judge on all samples."""
    from listqa_eval.llm_judge import HFJudgeModel, judge_list_elements

    logger.info("Loading judge model: %s", judge_model_name)
    judge = HFJudgeModel(
        model_name_or_path=judge_model_name,
        device=device,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )

    results = []
    for sample in tqdm(samples, desc="Evaluating with LLM judge"):
        question = sample["problem"]
        answer = sample["answer"]
        prediction = sample["prediction"]
        metadata = sample.get("metadata")

        # First get ROUGE-based list pairing
        list_pairs, num_pred, num_target = get_rouge_best_match_ranked_lists(
            answer, prediction, rouge_filter_threshold=0.0, rouge_score_type="rouge1"
        )

        # Run LLM judge on each pair
        element_scores = judge_list_elements(
            question=question,
            list_pairs=list_pairs,
            judge_model=judge,
            max_retries=max_retries,
        )

        result: Dict[str, Any] = {
            "question": question,
            "answer": answer,
            "prediction": prediction,
            "metadata": metadata,
            "recall_relaxed_em": compute_relaxed_recall_list_em(answer, prediction),
            "precision_relaxed_em": compute_relaxed_precision_list_em(answer, prediction),
            "llm_judge": {
                "list_element_scores": element_scores,
                "num_pred_list_elements": num_pred,
                "num_target_list_elements": num_target,
            },
        }

        if metadata:
            try:
                result["list_format_em"] = compute_list_format_em(prediction, metadata)
            except Exception:
                result["list_format_em"] = None

        results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="ListQA Evaluation — standalone benchmark for list-formatted QA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Data source (mutually exclusive)
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument(
        "--predictions",
        type=str,
        help="Path to JSONL file with predictions (must have 'problem', 'answer', 'prediction' fields)",
    )
    data_group.add_argument(
        "--hf-dataset",
        type=str,
        help="HuggingFace dataset name (requires 'prediction' column or --predictions-file)",
    )

    parser.add_argument("--hf-split", type=str, default="test", help="HF dataset split")
    parser.add_argument("--hf-subset", type=str, default=None, help="HF dataset subset/config")

    # Judge model
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="HuggingFace model name/path for LLM-as-a-Judge (e.g. 'meta-llama/Llama-3.1-8B-Instruct')",
    )
    parser.add_argument(
        "--no-llm-judge",
        action="store_true",
        help="Skip LLM judge; compute only ROUGE/EM metrics",
    )
    parser.add_argument("--device", type=str, default="auto", help="Device for judge model")
    parser.add_argument("--torch-dtype", type=str, default="auto",
                        choices=["auto", "float16", "bfloat16", "float32"],
                        help="Torch dtype for judge model")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="Trust remote code when loading the model")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Max retries per element for LLM judge")

    # Output
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save detailed results JSON")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum number of samples to evaluate (for debugging)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load data
    if args.predictions:
        logger.info("Loading predictions from %s", args.predictions)
        data = load_jsonl(args.predictions)
    else:
        logger.info("Loading dataset from HuggingFace: %s", args.hf_dataset)
        data = load_from_huggingface(args.hf_dataset, args.hf_split, args.hf_subset)

    validate_dataset(data, require_predictions=True)

    if args.max_samples:
        data = data[:args.max_samples]
        logger.info("Evaluating first %d samples", len(data))

    # Run evaluation
    if args.no_llm_judge or args.judge_model is None:
        if args.judge_model is None and not args.no_llm_judge:
            logger.warning("No --judge-model specified; running ROUGE-only evaluation. "
                          "Use --judge-model to enable LLM-as-a-Judge scoring.")

        logger.info("Running ROUGE/EM-only evaluation on %d samples...", len(data))
        results = []
        for sample in tqdm(data, desc="Computing metrics"):
            results.append(evaluate_sample_rouge_only(sample))

        # Aggregate ROUGE-only results
        n = len(results)
        avg_recall_em = sum(r["recall_relaxed_em"] for r in results) / n if n > 0 else 0
        avg_precision_em = sum(r["precision_relaxed_em"] for r in results) / n if n > 0 else 0
        avg_rouge_p = sum(r["rouge_list_precision"] for r in results) / n if n > 0 else 0
        avg_rouge_r = sum(r["rouge_list_recall"] for r in results) / n if n > 0 else 0
        avg_rouge_f1 = sum(r["rouge_list_f1"] for r in results) / n if n > 0 else 0

        summary = {
            "num_samples": n,
            "relaxed_em_recall": avg_recall_em,
            "relaxed_em_precision": avg_precision_em,
            "rouge_list_precision": avg_rouge_p,
            "rouge_list_recall": avg_rouge_r,
            "rouge_list_f1": avg_rouge_f1,
        }
    else:
        logger.info("Running full evaluation with LLM judge on %d samples...", len(data))
        results = evaluate_with_llm_judge(
            samples=data,
            judge_model_name=args.judge_model,
            device=args.device,
            torch_dtype=args.torch_dtype,
            max_retries=args.max_retries,
            trust_remote_code=args.trust_remote_code,
        )

        # Aggregate LLM judge scores
        judge_summary = aggregate_llm_judge_scores(results)

        # Also compute averages for EM metrics
        n = len(results)
        avg_recall_em = sum(r["recall_relaxed_em"] for r in results) / n if n > 0 else 0
        avg_precision_em = sum(r["precision_relaxed_em"] for r in results) / n if n > 0 else 0

        summary = {
            "num_samples": n,
            "relaxed_em_recall": avg_recall_em,
            "relaxed_em_precision": avg_precision_em,
            "llm_judge_scores": judge_summary,
        }

    # Print summary
    print("\n" + "=" * 60)
    print("ListQA Evaluation Results")
    print("=" * 60)
    print(json.dumps(summary, indent=2))
    print("=" * 60)

    # Save detailed results
    if args.output:
        output_path = Path(args.output)
        output_data = {"summary": summary, "per_sample_results": results}
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, default=str)
        logger.info("Detailed results saved to %s", output_path)

    return summary


if __name__ == "__main__":
    main()
