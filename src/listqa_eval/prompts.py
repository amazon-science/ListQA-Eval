"""
Prompt templates for ListQA evaluation.

Contains the system/user prompts for:
- Generating list-formatted answers (zero-shot and few-shot)
- LLM-as-a-Judge grading of individual list elements
"""

# ---------------------------------------------------------------------------
# Prompt for generating list answers (zero-shot)
# ---------------------------------------------------------------------------

LISTQA_ZERO_SHOT_PROMPT = (
    "Answer the question below by producing a list-formatted output, "
    "lists can be formatted one of two ways:\n\n"
    "For simple questions with few constraints requesting only main elements "
    "and not requesting sub-elements, use the following format:\n"
    "1. List Element 1\n"
    "2. List Element 2\n\n"
    "For complex questions asking for multiple factoids in addition to the "
    "primary question, use the following format:\n"
    "1. List Element 1\n"
    "* Category 1: Sub-Element 1\n"
    "* Category 2: Sub-Element 2\n\n"
    "2. List Element2\n"
    "* Category 1: Sub-Element 1\n"
    "* Category 2: Sub-Element 2\n\n"
    "All answers are bounded by 3 to 10 main elements, do not generate more "
    "than 10 main elements in the list. Do not add any information that is "
    "not asked for in the question.\n"
    "Question: {question}"
)


def format_listqa_prompt(question: str) -> str:
    """Format a zero-shot ListQA prompt for a given question."""
    return LISTQA_ZERO_SHOT_PROMPT.format(question=question)


# ---------------------------------------------------------------------------
# LLM-as-a-Judge prompt template
# ---------------------------------------------------------------------------

LLM_JUDGE_PROMPT_TEMPLATE = """\
Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT"]. The answer to the question is always going to be a list, however we will only be judging each list element one at a time, so the gold target and predicted answer may not completely answer the question and that is ok.
First, I will give examples of each grade, and then you will grade a new example.


The following are examples of CORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: malia obama
Predicted answer 1: malia
Predicted answer 2: Malia Obama
* birthday: July 4, 1998
Predicted answer 3: malia obama (born on July 4, 1998).
```

```
Question: What NFL teams has Tom Brady play for and what years was he active at these teams?
Gold target: New England Patriots
* Start Year: 2000
* End Year: 2019
Predicted answer 1: New England Patriots (2000 - 2019)
Predicted answer 2: patriots
* duration: 2000 - 2019
Predicted answer 3: NE Patriots
* Start: 2000
* End: 2019
```
These predicted answers are all CORRECT because:
    - They fully contain the important information in the gold target.
    - They do not contain any information that contradicts the gold target.
    - Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
    - Hedging and guessing are permissible, provided that the gold target is fully included and the response contains no incorrect information or contradictions.

The following are examples of INCORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia
Predicted answer 1: Susan
Predicted answer 2: Malia, Sasha, and Susan.
Predicted answer 3: Two children.
Predicted answer 4: Obama Family Tree
* Father: Barrack
* Mother: Michelle
* Child 1: Malia
* Child 2: Sasha
```

```
Question: What NFL teams has Tom Brady play for and what years was he active at these teams?
Gold target: New England Patriots
* Start Year: 2000
* End Year: 2019
Predicted answer 1: New England Patriots
Predicted answer 2: New England Patriots
* Start Year: 2002
* End Year: 2019
Predicted answer 3: Jacksonville Jaguars
Predicted answer 4: Jacksonville Jaguars
* Duration: 2000 - 2019
Predicted answer 5: New England Patriots
* Duration: 2000 - 2019
* Superbowl Wins: 7
* Touchdowns: 450
* Interceptions: 6
```
These predicted answers are all INCORRECT because:
    - A factual statement in the answer contradicts the gold target or does not provide all the requested information. Incorrect statements also provide information that cannot be validated against the gold target are also considered incorrect.

Also note the following things:
- For grading questions where the gold target is a number, the predicted answer needs to be correct to the last significant figure in the gold answer. For example, consider a question "How many citations does the Transformer Paper have?" with gold target "120k".
    - Predicted answers "120k", "124k", and 115k" are all CORRECT.
    - Predicted answers "100k" and "113k" are INCORRECT.
    - Predicted answers "around 100k" and "more than 50k" are considered INCORRECT because they neither confirm nor contradict the gold target.
- The gold target may contain more information than the question. In such cases, the predicted answer only needs to contain the information that is in the question.
    - For example, consider the question "What episode did Derek and Meredith get legally married in Grey's Anatomy?" with gold target "Season 7, Episode 20: White Wedding". Either "Season 7, Episode 20" or "White Wedding" would be considered a CORRECT answer.
- Do not punish predicted answers if they omit information that would be clearly inferred from the question.
    - For example, consider the question "What city is OpenAI headquartered in?" and the gold target "San Francisco, California". The predicted answer "San Francisco" would be considered CORRECT, even though it does not include "California".
    - Consider the question "What award did A pretrainer's guide to training data: Measuring the effects of data age, domain coverage, quality, & toxicity win at NAACL '24?", the gold target is "Outstanding Paper Award". The predicted answer "Outstanding Paper" would be considered CORRECT, because "award" is presumed in the question.
    - For the question "What is the height of Jason Wei in meters?", the gold target is "1.73 m". The predicted answer "1.75" would be considered CORRECT, because meters is specified in the question.
    - For the question "What is the name of Barack Obama's wife?", the gold target is "Michelle Obama". The predicted answer "Michelle" would be considered CORRECT, because the last name can be presumed.
- Do not punish for typos in people's name if it's clearly the same name.
    - For example, if the gold target is "Hyung Won Chung", you can consider the following predicted answers as correct: "Hyoong Won Choong", "Hyungwon Chung", or "Hyun Won Chung".

Here is a new example. Simply reply with either CORRECT, INCORRECT. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
```
Question: {question}
Gold target: {answer}
Predicted answer: {pred}
```

Grade the predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT

Just return the letters "A", or "B", with no text around it."""


def format_judge_prompt(question: str, gold_target: str, predicted: str) -> str:
    """Format the LLM judge prompt for a single list-element comparison."""
    return LLM_JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        answer=gold_target,
        pred=predicted,
    )
