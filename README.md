# ListQA-Eval

**Official evaluation code for [ListQA: A Benchmark for Evaluating List-Formatted Factual Knowledge Retrieval in Large Language Models](https://listqa.github.io) (NeurIPS 2026, Evaluations and Datasets Track).**

[[Project page]](https://listqa.github.io) · [[Dataset on Hugging Face]](https://huggingface.co/datasets/AmazonScience/ListQA) · [[Paper]](https://arxiv.org/pdf/2610.19165) · [[arXiv]](https://arxiv.org/abs/2610.19165)

>[!NOTE]
>This code is being released solely for academic and scientific reproducibility purposes, in support of the methods and findings described in the associated publication. Pull requests are not being accepted in order to maintain the code exactly as it was used in the paper.

ListQA measures how well language models produce structured, list-formatted answers to factoid questions. It evaluates both the **content accuracy** (are the list elements factually correct?) and **format compliance** (does the output follow the requested single-faceted or multi-faceted list structure?) using multiple complementary metrics — from lightweight string matching to full LLM-as-a-Judge grading.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [The ListQA Dataset](#the-listqa-dataset)
- [Dataset Format](#dataset-format)
- [Generating Predictions](#generating-predictions)
- [Quick Start](#quick-start)
- [Evaluation Metrics](#evaluation-metrics)
  - [Relaxed Exact Match](#1-relaxed-exact-match-precision--recall)
  - [ROUGE List Match](#2-rouge-list-element-match)
  - [List Format Compliance](#3-list-format-compliance)
  - [LLM-as-a-Judge](#4-llm-as-a-judge)
- [LLM-as-a-Judge Details](#llm-as-a-judge-details)
- [Python API](#python-api)
- [CLI Reference](#cli-reference)
- [Output Format](#output-format)
- [Running Tests](#running-tests)
- [Project Structure](#project-structure)
- [License](#license)
- [Citation](#citation)

---

## Overview

ListQA evaluates list-formatted question answering, where models must generate structured numbered lists in response to factoid questions. Questions come in two flavours:

- **Single-faceted** — simple lists where each element is a standalone fact:
  ```
  1. Alabama A&M
  2. Grambling State
  3. Savannah State
  4. Southeast Missouri State
  ```

- **Multi-faceted** — complex lists where each element contains sub-categories:
  ```
  1. 2018
  * Name of the work: "IDGAF"

  2. 2020
  * Name of the work: "Physical"
  ```

The evaluation pipeline:
1. **Parses** predicted and reference lists into individual elements
2. **Matches** predicted elements to reference elements using ROUGE + Hungarian assignment
3. **Grades** each matched pair using an LLM-as-a-Judge (locally via HuggingFace Transformers)
4. **Aggregates** scores into precision, recall, and F1 with optional metadata breakdowns

---

## Features

- **Zero proprietary dependencies** — completely standalone, no ties to any internal evaluation framework
- **Local LLM-as-a-Judge** via HuggingFace `transformers.AutoModelForCausalLM` — runs on CPU or GPU, no API keys or cloud services required
- **Multiple metric types**:
  - Relaxed exact-match precision & recall (substring containment)
  - ROUGE-based list element matching with optimal Hungarian assignment
  - List format compliance checking (single-faceted vs multi-faceted)
  - LLM-as-a-Judge element-level grading with aggregated P/R/F1
- **Flexible data loading** — local JSONL files or HuggingFace Hub datasets
- **Metadata-sliced breakdowns** by country, category, and year range
- **CLI and Python API** for easy integration into research workflows
- **Comprehensive test suite** — 56 tests with real dataset samples and mocked judge

---

## Installation

```bash
# From the repository
git clone https://github.com/amazon-science/ListQA-Eval.git
cd ListQA-Eval

# Option 1: Install as editable package (recommended)
pip install -e ".[dev]"

# Option 2: Install from requirements.txt
pip install -r requirements.txt
pip install -e .
```

### Requirements

| Dependency | Version | Purpose |
|---|---|---|
| Python | ≥ 3.9 | Runtime |
| PyTorch | ≥ 2.0 | LLM judge inference |
| HuggingFace Transformers | ≥ 4.35 | Model loading & generation |
| HuggingFace Accelerate | ≥ 0.24 | Device management |
| evaluate | ≥ 0.4 | ROUGE metric computation |
| rouge-score | ≥ 0.1.2 | ROUGE scoring backend |
| scipy | ≥ 1.10 | Hungarian assignment algorithm |
| Jinja2 | ≥ 3.1 | Prompt template rendering |
| datasets | ≥ 2.14 | HuggingFace Hub data loading |
| tqdm | ≥ 4.65 | Progress bars |

---

## The ListQA Dataset

The benchmark is hosted on Hugging Face at [`AmazonScience/ListQA`](https://huggingface.co/datasets/AmazonScience/ListQA). It has 9,045 human-curated questions grounded in Wikipedia tables, and every answer is a list of 3–10 elements.

| Config | Split(s) | Rows | Description |
|---|---|---|---|
| `all` (default) | `test` | 9,045 | Full benchmark |
| `simple` | `test` | 5,378 | Single-faceted questions (SimpleListQA) |
| `complex` | `test` | 3,667 | Multi-faceted questions (ComplexListQA) |
| `few_shot` | `simple` / `complex` / `combined` | 14 / 10 / 12 | In-context example pools for few-shot prompting |

```python
from datasets import load_dataset

ds = load_dataset("AmazonScience/ListQA", "simple", split="test")
```

The dataset has the questions and reference answers but no model outputs. You generate predictions first (see [Generating Predictions](#generating-predictions)) and then score them with this package.

---

## Dataset Format

### Input Data (JSONL)

Each line in the JSONL file must be a JSON object with these fields:

| Field | Required | Description |
|---|---|---|
| `problem` | ✅ | The question |
| `answer` | ✅ | Gold reference answer (list-formatted) |
| `prediction` | ✅ (for eval) | Model's generated answer |
| `metadata` | ❌ | JSON string with optional metadata |

### Metadata Schema

The optional `metadata` field is a JSON string that can contain:

```json
{
  "prompt_type": "single-faceted",  // or "multi-faceted"
  "category": "Sports",             // topic category
  "country": "USA",                 // country of origin
  "year_range": "2018-2019",        // temporal range
  "ordered": false,                 // whether list order matters
  "wikitable_source": "https://en.wikipedia.org/wiki/..."  // source Wikipedia page
}
```

`prompt_type` is required for the format-compliance metric. The other fields are only used to break down scores. Rows from the Hugging Face dataset already contain `metadata` in this form.

### Example Sample

```json
{
  "problem": "Which Division One teams were ineligible to participate in the 2018 NCAA tournament due to failing to meet APR requirements?",
  "answer": "Four Division One teams were ineligible...\n\n1. Alabama A&M\n2. Grambling State\n3. Savannah State\n4. Southeast Missouri State",
  "prediction": "1. Alabama A&M\n2. Grambling State\n3. Savannah State\n4. Southeast Missouri State",
  "metadata": "{\"category\": \"Sports\", \"year_range\": \"2018-2019\", \"prompt_type\": \"single-faceted\", \"country\": \"USA\"}"
}
```

### Multi-faceted Example

```json
{
  "problem": "Which years between 2017 and 2024 were works performed by Dua Lipa nominated for the MTV VMA for Best Choreography, and what was the name of the work?",
  "answer": "...\n\n1. 2018\n* Name of the work: \"IDGAF\"\n\n2. 2020\n* Name of the work: \"Physical\"\n\n3. 2023\n* Name of the work: \"Dance the Night\"\n\n4. 2024\n* Name of the work: \"Houdini\"",
  "prediction": "1. 2018\n* Name of the work: \"IDGAF\"\n\n2. 2020\n* Name of the work: \"Physical\"\n\n3. 2023\n* Name of the work: \"Dance the Night\"\n\n4. 2024\n* Name of the work: \"Houdini\"",
  "metadata": "{\"category\": \"Entertainment\", \"year_range\": \"2017-2024\", \"prompt_type\": \"multi-faceted\", \"country\": \"USA\"}"
}
```

---

## Generating Predictions

To build a predictions file, load the questions from the Hub, wrap each one in the zero-shot instruction prompt the paper uses, and write each row back out with a `prediction` field:

```python
import json
from datasets import load_dataset
from listqa_eval.prompts import format_listqa_prompt

ds = load_dataset("AmazonScience/ListQA", "all", split="test")

with open("predictions.jsonl", "w") as f:
    for row in ds:
        prompt = format_listqa_prompt(row["problem"])
        prediction = my_model.generate(prompt)  # replace with your model / API call
        f.write(json.dumps({**row, "prediction": prediction}) + "\n")
```

For few-shot evaluation, put 5 examples from the matching `few_shot` split ahead of the question: `simple` for the `simple` config, `complex` for `complex`, and `combined` for `all`.

---

## Quick Start

### 1. ROUGE-only evaluation (no GPU needed)

```bash
listqa-eval --predictions predictions.jsonl --no-llm-judge --output results.json
```

### 2. Full evaluation with LLM-as-a-Judge

```bash
listqa-eval \
  --predictions predictions.jsonl \
  --judge-model openai/gpt-oss-120b \
  --torch-dtype bfloat16 \
  --output results.json
```

### 3. Load from HuggingFace Hub

`--hf-dataset` takes any Hub dataset whose rows contain a `prediction` column, for example your own copy of ListQA with model outputs added. The official `AmazonScience/ListQA` dataset has no predictions, so pass it to `--hf-dataset` only after you have added them.

```bash
listqa-eval \
  --hf-dataset your-username/listqa-my-model-predictions \
  --hf-split test \
  --judge-model openai/gpt-oss-120b \
  --output results.json
```

### 4. Quick debug run (first 10 samples)

```bash
listqa-eval \
  --predictions predictions.jsonl \
  --judge-model openai/gpt-oss-120b \
  --max-samples 10 \
  --verbose \
  --output debug_results.json
```

---

## Evaluation Metrics

### 1. Relaxed Exact Match (Precision & Recall)

**How it works:** Checks whether each list element from the reference appears as a substring in the prediction (recall), and vice versa (precision). For multi-faceted elements, both the main element and all sub-elements are checked independently.

**When to use:** Fast, deterministic baseline metric. Useful for cases where the model output closely mirrors the reference text.

**Limitations:** Purely syntactic — misses semantic equivalences (e.g., "NE Patriots" vs "New England Patriots").

```python
from listqa_eval.metrics import compute_relaxed_recall_list_em, compute_relaxed_precision_list_em

recall = compute_relaxed_recall_list_em(reference, prediction)   # 0.0 to 1.0
precision = compute_relaxed_precision_list_em(reference, prediction)  # 0.0 to 1.0
```

### 2. ROUGE List Element Match

**How it works:**
1. Extracts numbered list elements from both reference and prediction
2. Computes pairwise ROUGE-1 scores between all prediction-reference pairs
3. Uses the **Hungarian algorithm** (optimal assignment) to find the best one-to-one matching that maximizes total ROUGE
4. Filters out matches below a configurable threshold (default: 0.0 for pairing, 0.5 for counting as correct)
5. Computes precision (correct/predicted) and recall (correct/target)

**When to use:** Better than exact match at handling paraphrases and word-order differences. Good intermediate metric between EM and LLM judge.

```python
from listqa_eval.metrics import compute_rouge_list_match

result = compute_rouge_list_match(reference, prediction)
# Returns: {"rouge_list_precision": 0.8, "rouge_list_recall": 0.75, "rouge_list_f1": 0.774, ...}
```

### 3. List Format Compliance

**How it works:** Checks whether the model follows the correct list format as specified in the metadata:
- **Single-faceted**: elements should NOT have `* Category: Value` sub-elements
- **Multi-faceted**: elements SHOULD have sub-elements
- Requires at least 3 list elements to be considered valid

**When to use:** Measures instruction-following ability for list formatting.

```python
from listqa_eval.metrics import compute_list_format_em

score = compute_list_format_em(prediction, metadata_json)  # 0.0 to 1.0
```

### 4. LLM-as-a-Judge

**How it works:**
1. Each predicted element is paired with its best-matching reference element (via ROUGE Hungarian assignment)
2. For each pair, a detailed prompt is constructed asking the judge to grade as **CORRECT** (A) or **INCORRECT** (B)
3. The judge prompt includes in-context examples of both correct and incorrect judgements
4. Unmatched predictions automatically receive grade B
5. Scores are aggregated into per-sample and overall precision/recall/F1

**When to use:** Most accurate metric. Captures semantic equivalence, handles typos, abbreviations, and partial information correctly. Required for final benchmark reporting.

```python
from listqa_eval.llm_judge import HFJudgeModel, judge_list_elements

judge = HFJudgeModel("openai/gpt-oss-120b", torch_dtype="bfloat16")
element_scores = judge_list_elements(question, list_pairs, judge)
```

---

## LLM-as-a-Judge Details

### Judge Model

Any HuggingFace causal language model can be used as the judge. Every result in the paper uses `openai/gpt-oss-120b`. The smaller judges below produce near-identical sample rankings (Spearman ρ > 0.96, mean absolute score difference ≤ 0.024):

| Model | Size | Notes |
|---|---|---|
| `openai/gpt-oss-120b` | 120B | Default judge used in the paper |
| `Qwen/Qwen3.5-27B` | 27B | Strong quality, good speed tradeoff |
| `Qwen/Qwen3.5-9B` | 9B | Good balance of quality and speed |
| `Qwen/Qwen3.5-4B` | 4B | Lightweight, fast alternative |

### Grading Criteria

The judge evaluates each element pair based on these principles:
- **Semantic meaning matters** — capitalization, punctuation, grammar, and order don't
- **Hedging/guessing is OK** if the gold target information is fully contained
- **Numbers** must be correct to the last significant figure in the gold answer
- **Omitted information** is OK if it would be clearly inferred from the question
- **Typos in names** are tolerated if clearly the same person/entity
- **Extra information** that cannot be validated against the gold target is considered **INCORRECT**

### Configuration

| Parameter | Default | Description |
|---|---|---|
| `--judge-model` | (required) | HuggingFace model name or local path |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, `cuda:0`, etc. |
| `--torch-dtype` | `auto` | `auto`, `float16`, `bfloat16`, `float32` |
| `--trust-remote-code` | `false` | Trust remote code in model repos |
| `--max-retries` | `3` | Retries per element on invalid judge output |

---

## Python API

### Computing Individual Metrics

```python
from listqa_eval.metrics import (
    compute_relaxed_recall_list_em,
    compute_relaxed_precision_list_em,
    compute_rouge_list_match,
    compute_list_format_em,
    get_rouge_best_match_ranked_lists,
)

target = "\n1. Alabama A&M\n2. Grambling State\n3. Savannah State"
pred = "\n1. Alabama A&M\n2. Grambling State\n3. Savannah State"

recall = compute_relaxed_recall_list_em(target, pred)
precision = compute_relaxed_precision_list_em(target, pred)
rouge = compute_rouge_list_match(target, pred)
print(f"EM Recall: {recall:.3f}, Precision: {precision:.3f}")
print(f"ROUGE P/R/F1: {rouge['rouge_list_precision']:.3f} / "
      f"{rouge['rouge_list_recall']:.3f} / {rouge['rouge_list_f1']:.3f}")
```

### Running LLM Judge

```python
from listqa_eval.llm_judge import HFJudgeModel, judge_list_elements
from listqa_eval.metrics import get_rouge_best_match_ranked_lists

judge = HFJudgeModel("openai/gpt-oss-120b", torch_dtype="bfloat16")

list_pairs, num_pred, num_target = get_rouge_best_match_ranked_lists(target, pred)
scores = judge_list_elements(question="What teams failed APR?", list_pairs=list_pairs, judge_model=judge)

for s in scores:
    print(f"  {s['score']}: {s['justification'][:80]}")
```

### Full Pipeline with Aggregation

```python
from listqa_eval.data import load_jsonl, validate_dataset
from listqa_eval.run import evaluate_with_llm_judge
from listqa_eval.aggregator import aggregate_llm_judge_scores

data = load_jsonl("predictions.jsonl")
validate_dataset(data, require_predictions=True)

results = evaluate_with_llm_judge(data, judge_model_name="openai/gpt-oss-120b")
summary = aggregate_llm_judge_scores(results)
print(f"Overall: P={summary['precision']:.3f} R={summary['recall']:.3f} F1={summary['f1']:.3f}")

if "category" in summary:
    for cat, scores in summary["category"].items():
        print(f"  {cat}: P={scores['precision']:.3f} R={scores['recall']:.3f} F1={scores['f1']:.3f}")
```

### Loading Data from HuggingFace

```python
from listqa_eval.data import load_from_huggingface

# Questions + references only (add a "prediction" field before evaluating)
data = load_from_huggingface("AmazonScience/ListQA", split="test", subset="simple")
```

---

## CLI Reference

```
listqa-eval --help

Usage: listqa-eval [OPTIONS]

Data Sources (mutually exclusive):
  --predictions PATH       Path to JSONL file with predictions
  --hf-dataset NAME        HuggingFace dataset identifier
  --hf-split SPLIT         Dataset split (default: test)
  --hf-subset SUBSET       Dataset subset/configuration name

Judge Model:
  --judge-model MODEL      HuggingFace model for LLM-as-a-Judge
  --no-llm-judge           Skip LLM judge; compute only ROUGE/EM metrics
  --device DEVICE           Device for judge model (auto/cpu/cuda)
  --torch-dtype DTYPE       Model precision (auto/float16/bfloat16/float32)
  --trust-remote-code       Trust remote code when loading the model
  --max-retries N           Max retries per element for LLM judge (default: 3)

Output:
  --output PATH             Path to save detailed results JSON
  --max-samples N           Maximum number of samples to evaluate
  --verbose                 Enable debug logging
```

---

## Output Format

### Summary (printed to stdout)

```json
{
  "num_samples": 100,
  "relaxed_em_recall": 0.72,
  "relaxed_em_precision": 0.68,
  "llm_judge_scores": {
    "num_samples": 100,
    "precision": 0.65,
    "recall": 0.61,
    "f1": 0.63,
    "category": {
      "Sports": {"num_samples": 30, "precision": 0.70, "recall": 0.65, "f1": 0.674},
      "Entertainment": {"num_samples": 25, "precision": 0.60, "recall": 0.58, "f1": 0.59}
    },
    "country": {
      "USA": {"num_samples": 80, "precision": 0.66, "recall": 0.62, "f1": 0.64}
    }
  }
}
```

### Detailed Results (saved with `--output`)

The output JSON contains both `summary` and `per_sample_results` with full per-element judge scores, prompts, and justifications for reproducibility.

---

## Running Tests

### Setup

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Or using requirements.txt
pip install -r requirements.txt
pip install -e .
```

### Run all tests

```bash
pytest tests/ -v
```

### Run specific test classes

```bash
# Only metrics tests
pytest tests/test_metrics.py::TestRelaxedRecallEMWithRealData -v

# Only mocked LLM judge tests
pytest tests/test_metrics.py::TestJudgeListElementsWithMock -v

# Only end-to-end pipeline tests
pytest tests/test_metrics.py::TestEndToEndPipeline -v

# Only aggregator tests
pytest tests/test_metrics.py::TestAggregatorWithRealScenarios -v
```

### Run with coverage

```bash
pytest tests/ -v --cov=listqa_eval --cov-report=term-missing
```

### Test categories

The test suite includes **56 tests** across these categories:

| Category | Tests | Description |
|---|---|---|
| `TestNormalizeStrings` / `TestNormalizeAnswers` | 5 | Text normalization edge cases |
| `TestParseReferenceListElements` | 4 | List parsing on real NCAA, Dua Lipa, Senate data |
| `TestParseMultiFacetedListElement` | 2 | Multi-faceted element parsing |
| `TestFlattenList` | 1 | Flattening multi-faceted to flat lists |
| `TestRelaxedRecallEMWithRealData` | 4 | Relaxed recall with perfect, partial, zero match |
| `TestRelaxedPrecisionEMWithRealData` | 3 | Relaxed precision with extra predictions |
| `TestListFormatEMWithRealData` | 4 | Format compliance for single/multi-faceted |
| `TestRougeListMatchWithRealData` | 5 | ROUGE + Hungarian assignment matching |
| `TestLLMJudgeParsing` | 7 | Response parsing (A, B, whitespace, multiline, invalid) |
| `TestJudgeListElementsWithMock` | 7 | Mocked judge: correct, mixed, retry, error, no-target |
| `TestAggregatorWithRealScenarios` | 5 | Aggregation with metadata breakdowns |
| `TestEndToEndPipeline` | 2 | Full pipeline: metrics → ROUGE → judge → aggregation |
| `TestDataLoading` | 5 | JSONL loading and validation |
| `TestPrompts` | 2 | Prompt template construction |

All tests use real data samples embedded inline from the ListQA dataset — **no external file access required**.

---

## Project Structure

```
ListQA-Eval/
├── pyproject.toml                  # Package config & dependencies
├── requirements.txt                # Pip requirements file
├── README.md                       # This file
├── LICENSE.md                      # CC BY-NC 4.0
├── src/listqa_eval/
│   ├── __init__.py                 # Package exports
│   ├── metrics.py                  # Core metrics (EM, ROUGE, format compliance)
│   ├── prompts.py                  # Prompt templates (QA generation + judge grading)
│   ├── llm_judge.py                # HuggingFace LLM-as-a-Judge implementation
│   ├── aggregator.py               # Score aggregation with metadata slicing
│   ├── data.py                     # JSONL + HuggingFace Hub data loading
│   └── run.py                      # CLI entry-point & evaluation pipeline
└── tests/
    └── test_metrics.py             # 56 comprehensive unit tests
```

---

## License

This code is released under the [Creative Commons Attribution-NonCommercial 4.0 International License](LICENSE.md) (CC BY-NC 4.0). The [ListQA dataset](https://huggingface.co/datasets/AmazonScience/ListQA) uses the same license.

---

## Citation

If you use ListQA or this evaluation code, please cite:

```bibtex
@misc{kulkarni2026listqabenchmarkevaluatinglistformatted,
      title={ListQA: A Benchmark for Evaluating List-Formatted Factual Knowledge Retrieval in Large Language Models}, 
      author={Mayank Kulkarni and Rakesh Vaideeswaran},
      year={2026},
      eprint={2610.19165},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2610.19165}, 
}
```
