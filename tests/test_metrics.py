"""
Comprehensive tests for listqa-eval using real dataset samples, simulated
model predictions, and mocked LLM-as-a-Judge responses.

All test data is embedded inline — no external file access required.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from listqa_eval.metrics import (
    normalize_strings,
    normalize_answers,
    parse_reference_list_elements,
    parse_multi_faceted_list_element,
    flatten_list,
    compute_relaxed_recall_list_em,
    compute_relaxed_precision_list_em,
    compute_list_format_em,
    compute_rouge_list_match,
    get_rouge_best_match_ranked_lists,
)
from listqa_eval.aggregator import aggregate_llm_judge_scores
from listqa_eval.llm_judge import _parse_judge_response, judge_list_elements
from listqa_eval.prompts import format_judge_prompt, format_listqa_prompt
from listqa_eval.data import load_jsonl, validate_dataset


# =====================================================================
# Real dataset samples (embedded inline from the ListQA dataset)
# =====================================================================

# --- Simple / single-faceted sample: NCAA teams ---
SIMPLE_SAMPLE = {
    "problem": "Which Division One teams were ineligible to participate in the 2018 NCAA tournament due to failing to meet APR requirements?",
    "answer": (
        "Four Division One teams were ineligible to participate in the 2018 NCAA tournament "
        "because they failed APR requirements, which is a measurement based upon how individual "
        "team members do academically. They were: \n\n"
        "1. Alabama A&M\n"
        "2. Grambling State\n"
        "3. Savannah State\n"
        "4. Southeast Missouri State"
    ),
    "metadata": json.dumps({
        "category": "Sports",
        "year_range": "2018-2019",
        "prompt_type": "single-faceted",
        "country": "USA",
    }),
}

# --- Complex / multi-faceted sample: Dua Lipa MTV nominations ---
COMPLEX_SAMPLE = {
    "problem": (
        "Which years between 2017 and 2024 were works performed by Dua Lipa nominated "
        "for the MTV Video Music Award for Best Choreography, and what was the name of "
        "the work performed by her?"
    ),
    "answer": (
        "Here is a list of the years between 2017 and 2024 where works performed by Dua Lipa "
        "were nominated for the MTV Video Music Award for Best Choreography, along with the "
        "name of the work performed by Dua Lipa:\n\n"
        "1. 2018\n* Name of the work: \"IDGAF\"\n\n"
        "2. 2020\n* Name of the work: \"Physical\"\n\n"
        "3. 2023\n* Name of the work: \"Dance the Night\"\n\n"
        "4. 2024\n* Name of the work: \"Houdini\""
    ),
    "metadata": json.dumps({
        "category": "Entertainment",
        "year_range": "2017-2024",
        "prompt_type": "multi-faceted",
        "country": "USA",
    }),
}

# --- Simple sample: Senate seats ---
SENATE_SAMPLE = {
    "problem": "How many Senate seats were controlled by the Democratic Party in each Class after the 2022 United States election cycle?",
    "answer": (
        "The United States Senate is divided into three classes. In total the Democratic Party "
        "held 48 seats in the senate after the 2022 United States election cycle. Here are how "
        "many Senate seats were held by members of the Democratic Party in each Class after the "
        "2022 United States Election Cycle:\n\n"
        "1. Class 1: 20 seats\n"
        "2. Class 2: 13 seats\n"
        "3. Class 3: 15 seats"
    ),
    "metadata": json.dumps({
        "category": "Politics",
        "year_range": "2022-2023",
        "prompt_type": "single-faceted",
        "country": "USA",
    }),
}


# =====================================================================
# Simulated model predictions
# =====================================================================

# Perfect prediction: model gets all elements correct
PERFECT_SIMPLE_PREDICTION = (
    "1. Alabama A&M\n"
    "2. Grambling State\n"
    "3. Savannah State\n"
    "4. Southeast Missouri State"
)

# Partial prediction: 2 of 4 correct, 1 wrong, 1 extra
PARTIAL_SIMPLE_PREDICTION = (
    "1. Alabama A&M\n"
    "2. Grambling State\n"
    "3. University of Memphis\n"
    "4. Florida State\n"
    "5. Kentucky Wildcats"
)

# Empty / non-list prediction
EMPTY_PREDICTION = "I'm not sure about this question."

# Perfect multi-faceted prediction for Dua Lipa
PERFECT_COMPLEX_PREDICTION = (
    "1. 2018\n* Name of the work: \"IDGAF\"\n\n"
    "2. 2020\n* Name of the work: \"Physical\"\n\n"
    "3. 2023\n* Name of the work: \"Dance the Night\"\n\n"
    "4. 2024\n* Name of the work: \"Houdini\""
)

# Partial complex prediction: correct years but wrong song names
PARTIAL_COMPLEX_PREDICTION = (
    "1. 2018\n* Name of the work: \"New Rules\"\n\n"
    "2. 2020\n* Name of the work: \"Physical\"\n\n"
    "3. 2022\n* Name of the work: \"Levitating\"\n\n"
    "4. 2024\n* Name of the work: \"Houdini\""
)

# Senate prediction: perfect match
PERFECT_SENATE_PREDICTION = (
    "1. Class 1: 20 seats\n"
    "2. Class 2: 13 seats\n"
    "3. Class 3: 15 seats"
)


# =====================================================================
# Text Normalisation Tests
# =====================================================================

class TestNormalizeStrings:
    def test_basic(self):
        assert normalize_strings("Hello \nWorld") == "hello \nworld"

    def test_strips_trailing_space_before_newline(self):
        # normalize_strings replaces "\n " with "\n", then lowercases
        assert normalize_strings("foo \n bar") == "foo \nbar"

    def test_invalid_input(self):
        with pytest.raises(ValueError):
            normalize_strings(123)


class TestNormalizeAnswers:
    def test_removes_after_example_4(self):
        text = "Some answer\nExample 4: blah"
        result = normalize_answers(text)
        assert "example 4" not in result
        assert result.startswith("\n")

    def test_no_example_4(self):
        text = "Just an answer"
        assert normalize_answers(text) == "\njust an answer"


# =====================================================================
# Parsing Tests with Real Data
# =====================================================================

class TestParseReferenceListElements:
    def test_simple_ncaa_answer(self):
        """Parse the real NCAA teams answer."""
        elements = parse_reference_list_elements(SIMPLE_SAMPLE["answer"])
        assert len(elements) == 4
        assert "alabama a&m" in elements[0]
        assert "grambling state" in elements[1]
        assert "savannah state" in elements[2]
        assert "southeast missouri state" in elements[3]

    def test_complex_dua_lipa_answer(self):
        """Parse the real Dua Lipa multi-faceted answer."""
        elements = parse_reference_list_elements(COMPLEX_SAMPLE["answer"])
        assert len(elements) == 4
        assert "2018" in elements[0]
        assert "idgaf" in elements[0]
        assert "2024" in elements[3]
        assert "houdini" in elements[3]

    def test_senate_seats_answer(self):
        """Parse the real Senate seats answer."""
        elements = parse_reference_list_elements(SENATE_SAMPLE["answer"])
        assert len(elements) == 3
        assert "class 1: 20 seats" in elements[0]

    def test_empty_string(self):
        assert parse_reference_list_elements("no list here") == []


class TestParseMultiFacetedListElement:
    def test_dua_lipa_element(self):
        """Parse a real multi-faceted element from Dua Lipa sample."""
        elem = '2018\n* Name of the work: "IDGAF"'
        result = parse_multi_faceted_list_element(elem)
        assert result["main_element"] == "2018"
        assert "Name of the work" in result["sub_elements"]
        assert '"IDGAF"' in result["sub_elements"]["Name of the work"]

    def test_simple_element(self):
        result = parse_multi_faceted_list_element("Alabama A&M")
        assert result["main_element"] == "Alabama A&M"
        assert result["sub_elements"] == {}


class TestFlattenList:
    def test_flatten_mixed(self):
        elements = ['2018\n* Name of the work: "IDGAF"', "Alabama A&M"]
        flat = flatten_list(elements)
        assert "2018" in flat
        assert '"IDGAF"' in flat
        assert "Alabama A&M" in flat


# =====================================================================
# Relaxed Exact Match Tests with Real Data
# =====================================================================

class TestRelaxedRecallEMWithRealData:
    def test_perfect_match_ncaa(self):
        score = compute_relaxed_recall_list_em(
            SIMPLE_SAMPLE["answer"], PERFECT_SIMPLE_PREDICTION
        )
        assert score == 1.0

    def test_partial_match_ncaa(self):
        score = compute_relaxed_recall_list_em(
            SIMPLE_SAMPLE["answer"], PARTIAL_SIMPLE_PREDICTION
        )
        # 2 of 4 teams correct (Alabama A&M, Grambling State)
        assert score == pytest.approx(0.5)

    def test_no_match(self):
        score = compute_relaxed_recall_list_em(
            SIMPLE_SAMPLE["answer"], EMPTY_PREDICTION
        )
        assert score == 0.0

    def test_perfect_match_senate(self):
        score = compute_relaxed_recall_list_em(
            SENATE_SAMPLE["answer"], PERFECT_SENATE_PREDICTION
        )
        assert score == 1.0


class TestRelaxedPrecisionEMWithRealData:
    def test_perfect_match_ncaa(self):
        score = compute_relaxed_precision_list_em(
            SIMPLE_SAMPLE["answer"], PERFECT_SIMPLE_PREDICTION
        )
        assert score == 1.0

    def test_partial_with_extra_predictions(self):
        score = compute_relaxed_precision_list_em(
            SIMPLE_SAMPLE["answer"], PARTIAL_SIMPLE_PREDICTION
        )
        # 2 correct out of 5 predictions
        assert score == pytest.approx(0.4)

    def test_empty_prediction(self):
        score = compute_relaxed_precision_list_em(
            SIMPLE_SAMPLE["answer"], EMPTY_PREDICTION
        )
        assert score == 0.0


# =====================================================================
# List Format Compliance Tests with Real Data
# =====================================================================

class TestListFormatEMWithRealData:
    def test_single_faceted_correct(self):
        """Simple prediction with no sub-elements should score 1.0 for single-faceted."""
        score = compute_list_format_em(
            PERFECT_SIMPLE_PREDICTION, SIMPLE_SAMPLE["metadata"]
        )
        assert score == 1.0

    def test_multi_faceted_correct(self):
        """Complex prediction with sub-elements should score 1.0 for multi-faceted."""
        score = compute_list_format_em(
            PERFECT_COMPLEX_PREDICTION, COMPLEX_SAMPLE["metadata"]
        )
        assert score == 1.0

    def test_wrong_format_single_as_multi(self):
        """Using multi-faceted format when single-faceted is expected."""
        score = compute_list_format_em(
            PERFECT_COMPLEX_PREDICTION, SIMPLE_SAMPLE["metadata"]
        )
        # The multi-faceted output has sub-elements; single-faceted expects none
        assert score == 0.0

    def test_too_few_elements(self):
        """Less than 3 elements should return 0."""
        short_pred = "1. Alabama A&M\n2. Grambling State"
        score = compute_list_format_em(short_pred, SIMPLE_SAMPLE["metadata"])
        assert score == 0.0


# =====================================================================
# ROUGE List Match Tests with Real Data
# =====================================================================

class TestRougeListMatchWithRealData:
    def test_perfect_match_ncaa(self):
        result = compute_rouge_list_match(
            SIMPLE_SAMPLE["answer"], PERFECT_SIMPLE_PREDICTION
        )
        assert result["rouge_list_precision"] == 1.0
        assert result["rouge_list_recall"] == 1.0
        assert result["rouge_list_f1"] == 1.0
        assert result["num_pred_list_elements"] == 4
        assert result["num_target_list_elements"] == 4

    def test_partial_match_has_lower_scores(self):
        result = compute_rouge_list_match(
            SIMPLE_SAMPLE["answer"], PARTIAL_SIMPLE_PREDICTION
        )
        assert result["rouge_list_precision"] < 1.0
        assert result["rouge_list_recall"] < 1.0
        assert result["num_pred_list_elements"] == 5
        assert result["num_target_list_elements"] == 4

    def test_empty_prediction(self):
        result = compute_rouge_list_match(
            SIMPLE_SAMPLE["answer"], EMPTY_PREDICTION
        )
        assert result["num_pred_list_elements"] == 0

    def test_perfect_match_senate(self):
        result = compute_rouge_list_match(
            SENATE_SAMPLE["answer"], PERFECT_SENATE_PREDICTION
        )
        assert result["rouge_list_precision"] == 1.0
        assert result["rouge_list_recall"] == 1.0

    def test_rouge_returns_list_pairs(self):
        """Verify list_pairs structure from ROUGE matching."""
        list_pairs, num_pred, num_target = get_rouge_best_match_ranked_lists(
            SIMPLE_SAMPLE["answer"], PERFECT_SIMPLE_PREDICTION
        )
        assert num_pred == 4
        assert num_target == 4
        assert len(list_pairs) == 4
        for pair in list_pairs:
            assert "pred" in pair
            assert "target" in pair
            assert "rouge_score" in pair
            assert pair["target"] is not None  # All should be matched


# =====================================================================
# LLM Judge Response Parsing Tests
# =====================================================================

class TestLLMJudgeParsing:
    def test_parse_A(self):
        assert _parse_judge_response("A") == "A"

    def test_parse_B(self):
        assert _parse_judge_response("B") == "B"

    def test_parse_with_whitespace(self):
        assert _parse_judge_response("  A  ") == "A"

    def test_parse_multiline_A(self):
        assert _parse_judge_response("The answer is:\nA\n") == "A"

    def test_parse_multiline_B(self):
        assert _parse_judge_response("Based on my analysis:\nB\n") == "B"

    def test_parse_invalid_verbose(self):
        assert _parse_judge_response("I think this is correct because...") == "[invalid_judgement]"

    def test_parse_empty(self):
        assert _parse_judge_response("") == "[invalid_judgement]"


# =====================================================================
# Mocked LLM-as-a-Judge Tests
# =====================================================================

class TestJudgeListElementsWithMock:
    """Tests for judge_list_elements using a mocked HFJudgeModel."""

    def _make_mock_judge(self, responses):
        """Create a mock judge that returns predefined responses."""
        mock_judge = MagicMock()
        mock_judge.generate = MagicMock(side_effect=responses)
        return mock_judge

    def test_all_correct_judgements(self):
        """Mock judge says A (CORRECT) for all 4 NCAA elements."""
        list_pairs = [
            {"pred": "alabama a&m", "target": "alabama a&m", "rouge_score": 1.0},
            {"pred": "grambling state", "target": "grambling state", "rouge_score": 1.0},
            {"pred": "savannah state", "target": "savannah state", "rouge_score": 1.0},
            {"pred": "southeast missouri state", "target": "southeast missouri state", "rouge_score": 1.0},
        ]
        mock_judge = self._make_mock_judge(["A", "A", "A", "A"])

        results = judge_list_elements(
            question=SIMPLE_SAMPLE["problem"],
            list_pairs=list_pairs,
            judge_model=mock_judge,
            max_retries=1,
        )

        assert len(results) == 4
        assert all(r["score"] == "A" for r in results)
        assert mock_judge.generate.call_count == 4

    def test_mixed_judgements(self):
        """Mock judge gives mixed A/B scores."""
        list_pairs = [
            {"pred": "alabama a&m", "target": "alabama a&m", "rouge_score": 1.0},
            {"pred": "university of memphis", "target": "grambling state", "rouge_score": 0.2},
            {"pred": "savannah state", "target": "savannah state", "rouge_score": 1.0},
            {"pred": "florida state", "target": "southeast missouri state", "rouge_score": 0.1},
        ]
        mock_judge = self._make_mock_judge(["A", "B", "A", "B"])

        results = judge_list_elements(
            question=SIMPLE_SAMPLE["problem"],
            list_pairs=list_pairs,
            judge_model=mock_judge,
            max_retries=1,
        )

        assert len(results) == 4
        assert results[0]["score"] == "A"
        assert results[1]["score"] == "B"
        assert results[2]["score"] == "A"
        assert results[3]["score"] == "B"

    def test_no_target_gets_automatic_B(self):
        """When target is None (no ROUGE match), score should be B without calling judge."""
        list_pairs = [
            {"pred": "some random team", "target": None, "rouge_score": None},
        ]
        mock_judge = self._make_mock_judge([])  # should not be called

        results = judge_list_elements(
            question=SIMPLE_SAMPLE["problem"],
            list_pairs=list_pairs,
            judge_model=mock_judge,
            max_retries=1,
        )

        assert len(results) == 1
        assert results[0]["score"] == "B"
        assert results[0]["justification"] == "No matching target element"
        assert mock_judge.generate.call_count == 0

    def test_retry_on_invalid_response(self):
        """Judge should retry when getting invalid responses."""
        list_pairs = [
            {"pred": "alabama a&m", "target": "alabama a&m", "rouge_score": 1.0},
        ]
        # First two responses are invalid, third is valid
        mock_judge = self._make_mock_judge(["hmm maybe correct", "not sure", "A"])

        results = judge_list_elements(
            question=SIMPLE_SAMPLE["problem"],
            list_pairs=list_pairs,
            judge_model=mock_judge,
            max_retries=3,
        )

        assert len(results) == 1
        assert results[0]["score"] == "A"
        assert mock_judge.generate.call_count == 3

    def test_multi_faceted_judge_calls(self):
        """Test judging multi-faceted elements from the Dua Lipa sample."""
        list_pairs = [
            {"pred": '2018\n* name of the work: "idgaf"', "target": '2018\n* name of the work: "idgaf"', "rouge_score": 1.0},
            {"pred": '2020\n* name of the work: "physical"', "target": '2020\n* name of the work: "physical"', "rouge_score": 1.0},
            {"pred": '2023\n* name of the work: "dance the night"', "target": '2023\n* name of the work: "dance the night"', "rouge_score": 1.0},
            {"pred": '2024\n* name of the work: "houdini"', "target": '2024\n* name of the work: "houdini"', "rouge_score": 1.0},
        ]
        mock_judge = self._make_mock_judge(["A", "A", "A", "A"])

        results = judge_list_elements(
            question=COMPLEX_SAMPLE["problem"],
            list_pairs=list_pairs,
            judge_model=mock_judge,
            max_retries=1,
        )

        assert len(results) == 4
        assert all(r["score"] == "A" for r in results)

    def test_judge_error_handling(self):
        """Judge should handle exceptions gracefully and return B."""
        list_pairs = [
            {"pred": "alabama a&m", "target": "alabama a&m", "rouge_score": 1.0},
        ]
        mock_judge = MagicMock()
        mock_judge.generate = MagicMock(side_effect=RuntimeError("GPU OOM"))

        results = judge_list_elements(
            question=SIMPLE_SAMPLE["problem"],
            list_pairs=list_pairs,
            judge_model=mock_judge,
            max_retries=2,
        )

        assert len(results) == 1
        assert results[0]["score"] == "B"  # defaults to B on error

    def test_judge_prompt_contains_question_and_target(self):
        """Verify the judge prompt is constructed correctly."""
        list_pairs = [
            {"pred": "grambling state", "target": "grambling state", "rouge_score": 1.0},
        ]
        mock_judge = self._make_mock_judge(["A"])

        results = judge_list_elements(
            question=SIMPLE_SAMPLE["problem"],
            list_pairs=list_pairs,
            judge_model=mock_judge,
            max_retries=1,
        )

        # Check that the prompt was constructed with the right data
        call_args = mock_judge.generate.call_args[0][0]
        assert "grambling state" in call_args.lower()
        assert "ncaa" in call_args.lower() or "division one" in call_args.lower()


# =====================================================================
# Aggregation Tests with Real-Scenario Data
# =====================================================================

class TestAggregatorWithRealScenarios:
    def test_aggregate_empty(self):
        result = aggregate_llm_judge_scores([])
        assert result["precision"] == 0.0
        assert result["num_samples"] == 0

    def test_perfect_simple_sample(self):
        """All 4 NCAA teams graded CORRECT."""
        results = [{
            "question": SIMPLE_SAMPLE["problem"],
            "answer": SIMPLE_SAMPLE["answer"],
            "prediction": PERFECT_SIMPLE_PREDICTION,
            "metadata": SIMPLE_SAMPLE["metadata"],
            "llm_judge": {
                "list_element_scores": [
                    {"score": "A"}, {"score": "A"}, {"score": "A"}, {"score": "A"},
                ],
                "num_pred_list_elements": 4,
                "num_target_list_elements": 4,
            },
        }]
        agg = aggregate_llm_judge_scores(results)
        assert agg["precision"] == 1.0
        assert agg["recall"] == 1.0
        assert agg["f1"] == 1.0
        assert agg["num_samples"] == 1
        # Check metadata breakdowns
        assert "category" in agg
        assert "Sports" in agg["category"]
        assert agg["category"]["Sports"]["precision"] == 1.0

    def test_partial_match_scoring(self):
        """2 of 5 predictions correct, 2 of 4 targets found."""
        results = [{
            "question": SIMPLE_SAMPLE["problem"],
            "answer": SIMPLE_SAMPLE["answer"],
            "prediction": PARTIAL_SIMPLE_PREDICTION,
            "metadata": SIMPLE_SAMPLE["metadata"],
            "llm_judge": {
                "list_element_scores": [
                    {"score": "A"}, {"score": "A"}, {"score": "B"},
                    {"score": "B"}, {"score": "B"},
                ],
                "num_pred_list_elements": 5,
                "num_target_list_elements": 4,
            },
        }]
        agg = aggregate_llm_judge_scores(results)
        assert agg["precision"] == pytest.approx(0.4)  # 2/5
        assert agg["recall"] == pytest.approx(0.5)  # 2/4
        assert agg["num_samples"] == 1

    def test_multi_sample_aggregation(self):
        """Aggregate across both simple and complex samples."""
        results = [
            {
                "question": SIMPLE_SAMPLE["problem"],
                "answer": SIMPLE_SAMPLE["answer"],
                "prediction": PERFECT_SIMPLE_PREDICTION,
                "metadata": SIMPLE_SAMPLE["metadata"],
                "llm_judge": {
                    "list_element_scores": [
                        {"score": "A"}, {"score": "A"}, {"score": "A"}, {"score": "A"},
                    ],
                    "num_pred_list_elements": 4,
                    "num_target_list_elements": 4,
                },
            },
            {
                "question": COMPLEX_SAMPLE["problem"],
                "answer": COMPLEX_SAMPLE["answer"],
                "prediction": PARTIAL_COMPLEX_PREDICTION,
                "metadata": COMPLEX_SAMPLE["metadata"],
                "llm_judge": {
                    "list_element_scores": [
                        {"score": "B"}, {"score": "A"}, {"score": "B"}, {"score": "A"},
                    ],
                    "num_pred_list_elements": 4,
                    "num_target_list_elements": 4,
                },
            },
        ]
        agg = aggregate_llm_judge_scores(results)
        assert agg["num_samples"] == 2
        # Sample 1: P=1.0, R=1.0; Sample 2: P=0.5, R=0.5
        assert agg["precision"] == pytest.approx(0.75)  # (1.0+0.5)/2
        assert agg["recall"] == pytest.approx(0.75)
        # Check category breakdown
        assert "category" in agg
        assert "Sports" in agg["category"]
        assert "Entertainment" in agg["category"]
        assert agg["category"]["Sports"]["precision"] == 1.0
        assert agg["category"]["Entertainment"]["precision"] == 0.5

    def test_year_range_and_country_breakdown(self):
        """Verify year_range and country slicing."""
        results = [
            {
                "question": SENATE_SAMPLE["problem"],
                "answer": SENATE_SAMPLE["answer"],
                "prediction": PERFECT_SENATE_PREDICTION,
                "metadata": SENATE_SAMPLE["metadata"],
                "llm_judge": {
                    "list_element_scores": [
                        {"score": "A"}, {"score": "A"}, {"score": "A"},
                    ],
                    "num_pred_list_elements": 3,
                    "num_target_list_elements": 3,
                },
            },
        ]
        agg = aggregate_llm_judge_scores(results)
        assert "country" in agg
        assert "USA" in agg["country"]
        assert agg["country"]["USA"]["f1"] == 1.0
        assert "year_range" in agg
        assert "2022-2023" in agg["year_range"]


# =====================================================================
# End-to-End Pipeline Test with Mocked Judge
# =====================================================================

class TestEndToEndPipeline:
    """Full pipeline: real data → metrics → mocked judge → aggregation."""

    def test_full_pipeline_simple_sample(self):
        """Run the complete evaluation pipeline on a simple sample."""
        sample = {**SIMPLE_SAMPLE, "prediction": PERFECT_SIMPLE_PREDICTION}

        # Step 1: Compute relaxed EM
        recall_em = compute_relaxed_recall_list_em(sample["answer"], sample["prediction"])
        precision_em = compute_relaxed_precision_list_em(sample["answer"], sample["prediction"])
        assert recall_em == 1.0
        assert precision_em == 1.0

        # Step 2: Compute format EM
        format_em = compute_list_format_em(sample["prediction"], sample["metadata"])
        assert format_em == 1.0

        # Step 3: ROUGE matching
        list_pairs, num_pred, num_target = get_rouge_best_match_ranked_lists(
            sample["answer"], sample["prediction"]
        )
        assert num_pred == 4
        assert num_target == 4

        # Step 4: Mock judge
        mock_judge = MagicMock()
        mock_judge.generate = MagicMock(return_value="A")

        element_scores = judge_list_elements(
            question=sample["problem"],
            list_pairs=list_pairs,
            judge_model=mock_judge,
            max_retries=1,
        )
        assert all(s["score"] == "A" for s in element_scores)

        # Step 5: Aggregate
        results = [{
            "question": sample["problem"],
            "answer": sample["answer"],
            "prediction": sample["prediction"],
            "metadata": sample["metadata"],
            "llm_judge": {
                "list_element_scores": element_scores,
                "num_pred_list_elements": num_pred,
                "num_target_list_elements": num_target,
            },
        }]
        summary = aggregate_llm_judge_scores(results)
        assert summary["precision"] == 1.0
        assert summary["recall"] == 1.0
        assert summary["f1"] == 1.0

    def test_full_pipeline_partial_complex_sample(self):
        """Pipeline with partially correct complex predictions."""
        sample = {**COMPLEX_SAMPLE, "prediction": PARTIAL_COMPLEX_PREDICTION}

        # ROUGE matching
        list_pairs, num_pred, num_target = get_rouge_best_match_ranked_lists(
            sample["answer"], sample["prediction"]
        )

        # Mock judge: Physical correct, Houdini correct, New Rules wrong, Levitating wrong
        responses = []
        for pair in list_pairs:
            if pair["target"] is None:
                continue
            pred = pair["pred"]
            if "physical" in pred or "houdini" in pred:
                responses.append("A")
            else:
                responses.append("B")

        mock_judge = MagicMock()
        mock_judge.generate = MagicMock(side_effect=responses)

        element_scores = judge_list_elements(
            question=sample["problem"],
            list_pairs=list_pairs,
            judge_model=mock_judge,
            max_retries=1,
        )

        # Count correct
        correct_count = sum(1 for s in element_scores if s["score"] == "A")
        assert correct_count == 2

        # Aggregate
        results = [{
            "question": sample["problem"],
            "answer": sample["answer"],
            "prediction": sample["prediction"],
            "metadata": sample["metadata"],
            "llm_judge": {
                "list_element_scores": element_scores,
                "num_pred_list_elements": num_pred,
                "num_target_list_elements": num_target,
            },
        }]
        summary = aggregate_llm_judge_scores(results)
        assert summary["precision"] == pytest.approx(0.5)
        assert summary["recall"] == pytest.approx(0.5)


# =====================================================================
# Data Loading Tests
# =====================================================================

class TestDataLoading:
    def test_load_jsonl(self, tmp_path):
        """Write real-format data to JSONL and reload."""
        f = tmp_path / "test.jsonl"
        lines = [
            json.dumps(SIMPLE_SAMPLE),
            json.dumps(COMPLEX_SAMPLE),
            json.dumps(SENATE_SAMPLE),
        ]
        f.write_text("\n".join(lines) + "\n")
        data = load_jsonl(f)
        assert len(data) == 3
        assert data[0]["problem"] == SIMPLE_SAMPLE["problem"]
        assert data[1]["problem"] == COMPLEX_SAMPLE["problem"]

    def test_load_jsonl_with_predictions(self, tmp_path):
        """Write data with predictions and validate."""
        f = tmp_path / "preds.jsonl"
        sample_with_pred = {**SIMPLE_SAMPLE, "prediction": PERFECT_SIMPLE_PREDICTION}
        f.write_text(json.dumps(sample_with_pred) + "\n")
        data = load_jsonl(f)
        validate_dataset(data, require_predictions=True)
        assert data[0]["prediction"] == PERFECT_SIMPLE_PREDICTION

    def test_validate_dataset_missing_field(self):
        with pytest.raises(ValueError, match="missing required fields"):
            validate_dataset([{"problem": "Q1"}])

    def test_validate_dataset_ok(self):
        validate_dataset([{"problem": "Q1", "answer": "A1"}])

    def test_validate_dataset_missing_prediction(self):
        with pytest.raises(ValueError, match="prediction"):
            validate_dataset(
                [{"problem": "Q1", "answer": "A1"}],
                require_predictions=True,
            )


# =====================================================================
# Prompt Template Tests
# =====================================================================

class TestPrompts:
    def test_format_listqa_prompt(self):
        prompt = format_listqa_prompt(SIMPLE_SAMPLE["problem"])
        assert SIMPLE_SAMPLE["problem"] in prompt
        assert "list-formatted output" in prompt
        assert "3 to 10 main elements" in prompt

    def test_format_judge_prompt_with_real_data(self):
        prompt = format_judge_prompt(
            question=SIMPLE_SAMPLE["problem"],
            gold_target="Alabama A&M",
            predicted="Alabama A&M University",
        )
        assert "Alabama A&M" in prompt
        assert "Alabama A&M University" in prompt
        assert "CORRECT" in prompt
        assert "INCORRECT" in prompt
        assert "Grade the predicted answer" in prompt
