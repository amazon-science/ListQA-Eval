"""
LLM-as-a-Judge for ListQA evaluation using HuggingFace Transformers.

Loads a model locally via `transformers.AutoModelForCausalLM` and scores
each predicted list element against its matched reference element.
No external API calls — runs entirely on local hardware (CPU or GPU).
"""

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from listqa_eval.prompts import format_judge_prompt

logger = logging.getLogger(__name__)


class HFJudgeModel:
    """
    Wrapper around a HuggingFace causal-LM used as a judge.

    Args:
        model_name_or_path: HuggingFace model identifier or local path
            (e.g. ``"meta-llama/Llama-3.1-8B-Instruct"``).
        device: ``"auto"``, ``"cpu"``, ``"cuda"``, ``"cuda:0"``, etc.
        torch_dtype: Data type for model weights (``"auto"``, ``"float16"``,
            ``"bfloat16"``, ``"float32"``).
        max_new_tokens: Maximum number of new tokens to generate per call.
        trust_remote_code: Whether to trust remote code in model repos.
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "auto",
        torch_dtype: str = "auto",
        max_new_tokens: int = 8,
        trust_remote_code: bool = False,
    ):
        self.model_name_or_path = model_name_or_path
        self.max_new_tokens = max_new_tokens

        # Resolve torch dtype
        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        resolved_dtype = dtype_map.get(torch_dtype, "auto")

        logger.info("Loading tokenizer from %s", model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("Loading model from %s (dtype=%s, device=%s)", model_name_or_path, torch_dtype, device)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=resolved_dtype,
            device_map=device,
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()

    def generate(self, prompt: str) -> str:
        """Generate text for a single prompt and return the raw output string."""
        # Try to use chat template if available
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = prompt

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode only the generated portion
        generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def _parse_judge_response(response: str) -> str:
    """
    Extract the grade (A or B) from the judge model's response.
    Falls back to ``"B"`` (INCORRECT) if the response cannot be parsed.
    """
    match = re.search(r"^\s*(A|B)\s*$", response, re.MULTILINE)
    if match:
        return match.group(1)
    # Fallback: look for A or B anywhere in a short response
    clean = response.strip()
    if clean in ("A", "B"):
        return clean
    return "[invalid_judgement]"


def judge_list_elements(
    question: str,
    list_pairs: List[Dict[str, Any]],
    judge_model: HFJudgeModel,
    max_retries: int = 3,
) -> List[Dict[str, Any]]:
    """
    Judge each (predicted, reference) list-element pair using the local LLM.

    Args:
        question: The original question.
        list_pairs: List of dicts with ``"pred"``, ``"target"``, ``"rouge_score"`` keys
            (as returned by ``get_rouge_best_match_ranked_lists``).
        judge_model: A loaded ``HFJudgeModel`` instance.
        max_retries: Number of retries per element on invalid output.

    Returns:
        List of dicts with ``"score"`` (``"A"`` or ``"B"``), ``"justification"``,
        and ``"judge_prompt"`` for each element.
    """
    results: List[Dict[str, Any]] = []

    for pair in list_pairs:
        if pair["target"] is None:
            results.append({
                "score": "B",
                "justification": "No matching target element",
                "judge_prompt": f"Did not run LLM as judge for: {pair['pred']}",
            })
            continue

        prompt = format_judge_prompt(
            question=question,
            gold_target=pair["target"],
            predicted=pair["pred"],
        )

        score = ""
        justification = ""
        for _ in range(max_retries):
            try:
                response = judge_model.generate(prompt)
                score = _parse_judge_response(response)
                justification = response
                if score in ("A", "B"):
                    break
            except Exception as e:
                logger.warning("Judge inference error: %s", e)
                score = "B"
                justification = f"Error: {e}"

        if score not in ("A", "B"):
            score = "B"

        results.append({
            "score": score,
            "justification": justification,
            "judge_prompt": prompt,
        })

    return results
