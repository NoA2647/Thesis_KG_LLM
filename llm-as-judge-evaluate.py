#!/usr/bin/env python3
"""
llm-as-judge-evaluate.py
LLM-as-judge evaluation of KGQA system output.
Fully independent from hit1-evaluate.py.

Gold files   : data/cwq.json, data/qald.json, data/webq.json
Output files : data/cwq_output.json, data/qald_output.json, data/webq_output.json
Results      : eval_results/{dataset}_llm_judge.json
Error log    : eval_results/llm_judge_errors.log

LLM model and API key are read from .env  (API_KEY, LLM_MODEL).
Predictions of "No Answer" / None are counted as wrong without calling the API.
"""

import json
import os
import time
import logging
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

# ── configuration ────────────────────────────────────────────────────────
API_KEY   = os.getenv("API_KEY",   "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

DATA_DIR    = Path("data")
RESULTS_DIR = Path("eval_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)   # must exist before logging setup

DATASETS: dict[str, dict] = {
    "cwq": {
        "gold":       DATA_DIR / "cwq.json",
        "output":     DATA_DIR / "cwq_output.json",
        "id_key":     "ID",
        "answer_key": "answer",   # single string
        "match_by":   "id",
    },
    "qald": {
        "gold":       DATA_DIR / "qald_10-en.json",
        "output":     DATA_DIR / "qald_output.json",
        "id_key":     None,
        "answer_key": "answer",   # dict {qid: label}
        "match_by":   "question",
    },
    "webq": {
        "gold":       DATA_DIR / "webquestions.json",
        "output":     DATA_DIR / "webquestions_output.json",
        "id_key":     None,
        "answer_key": "answers",  # list of strings
        "match_by":   "question",
    },
}

# Predictions that mean "I don't know" — count as 0 without an API call
NO_ANSWER_STRINGS: frozenset[str] = frozenset({"no answer", "none", "n/a", ""})

JUDGE_PROMPT = """\
You are evaluating a question answering system.
Question: {question}
Gold answer: {gold_answer}
Predicted answer: {predicted_answer}

Consider the predicted answer CORRECT if any of these apply:
1. It exactly matches the gold answer
2. It contains the gold answer as a substring (e.g., gold: "China", predicted: "Guangdong Province, China")
3. It is a more specific subcategory/type of the gold answer AND is accurate (e.g., gold: "Parliamentary system", predicted: "Parliamentary republic" - since a parliamentary republic IS a type of parliamentary system)
4. It is a valid alias or semantic reordering of the gold answer
5. Both answers express the same core concept but use different terminology

For hierarchical/categorical questions: If the predicted answer is a more specific valid instance \
or subtype of the gold answer, consider it correct as long as it's accurate and more informative.

Answer only YES or NO."""

# ── logging ──────────────────────────────────────────────────────────────
logging.basicConfig(
    filename=str(RESULTS_DIR / "llm_judge_errors.log"),
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)


# ── answer extraction from output items ──────────────────────────────────

def extract_predicted(item: dict) -> Optional[str]:
    """
    Handle two pipeline output shapes:
      • Normal:  item["answer"]["answer"]["final_answer"]  (dict with LLM reasoning)
      • CoT:     item["answer"]["answer"]                  (plain string)
    """
    answer_field = item.get("answer", {})
    if not isinstance(answer_field, dict):
        return None
    inner = answer_field.get("answer")
    if inner is None:
        return None
    if isinstance(inner, str):
        return inner.strip() or None
    if isinstance(inner, dict):
        fa = inner.get("final_answer")
        return str(fa).strip() if fa is not None else None
    return None


def extract_gold_list(gold_item: dict, answer_key: str) -> list[str]:
    """Normalise any gold answer shape into a flat list of strings."""
    val = gold_item.get(answer_key)
    if val is None:
        return []
    if isinstance(val, str):
        return [val]
    if isinstance(val, list):
        return [str(v) for v in val if v]
    if isinstance(val, dict):       # QALD: {qid: label}
        return [str(v) for v in val.values() if v]
    return [str(val)]


def is_no_answer(pred: Optional[str]) -> bool:
    return pred is None or pred.strip().lower() in NO_ANSWER_STRINGS


# ── LLM judge call with retry ─────────────────────────────────────────────

def call_judge(
    client:      OpenAI,
    question:    str,
    gold_str:    str,
    pred:        str,
    max_retries: int   = 3,
    retry_delay: float = 2.0,
) -> Optional[int]:
    """
    Ask the LLM whether pred matches gold.
    Returns 1 (YES), 0 (NO), or None on persistent API failure.
    None is excluded from the final average; the question is still logged.
    """
    prompt = JUDGE_PROMPT.format(
        question=question,
        gold_answer=gold_str,
        predicted_answer=pred,
    )
    for attempt in range(max_retries):
        try:
            rsp  = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=5,
            )
            text = rsp.choices[0].message.content.strip().upper()
            if "YES" in text:
                return 1
            if "NO" in text:
                return 0
            # Unexpected response — treat as NO and warn
            logger.warning("Unexpected judge response %r (attempt %d)", text, attempt + 1)
            return 0
        except Exception as exc:
            if attempt < max_retries - 1:
                logger.warning(
                    "Judge API attempt %d/%d failed: %s — retrying in %.1fs",
                    attempt + 1, max_retries, exc, retry_delay,
                )
                time.sleep(retry_delay)
            else:
                logger.error(
                    "Judge API failed after %d attempts: %s",
                    max_retries, exc, exc_info=True,
                )
    return None     # all retries exhausted


# ── per-dataset evaluation ────────────────────────────────────────────────

def evaluate_dataset(name: str, cfg: dict, client: OpenAI) -> dict:
    with open(cfg["gold"],   encoding="utf-8") as f:
        gold_data: list[dict] = json.load(f)
    with open(cfg["output"], encoding="utf-8") as f:
        output_data: list[dict] = json.load(f)

    # build lookup: key → gold item
    if cfg["match_by"] == "id":
        gold_lookup = {str(item[cfg["id_key"]]): item for item in gold_data}
    else:
        # exact question string match (as per spec)
        gold_lookup = {item["question"]: item for item in gold_data}

    per_question: list[dict] = []
    judge_scores: list[int]  = []   # 0 or 1; None scores are excluded
    total = 0

    for item in tqdm(output_data, desc=f"  judge / {name}", unit="q", ncols=82):
        total += 1
        q_id     = str(item.get("id", ""))
        question = item.get("question", "")

        row: dict = {
            "id":             q_id,
            "question":       question,
            "predicted":      None,
            "gold":           [],
            "hits1_strict":   None,     # not computed by this script
            "hits1_flexible": None,
            "matched_layer":  None,
            "llm_judge":      None,
        }

        try:
            lookup_key = q_id if cfg["match_by"] == "id" else question
            gold_item  = gold_lookup.get(lookup_key)

            if gold_item is None:
                logger.warning(
                    "[%s] no gold match  id=%r  q=%.60r", name, q_id, question
                )
                per_question.append(row)
                continue

            pred  = extract_predicted(item)
            golds = extract_gold_list(gold_item, cfg["answer_key"])
            row.update(predicted=pred, gold=golds)

            # ── no-answer shortcut: wrong, no API call ──────────────────
            if is_no_answer(pred):
                row["llm_judge"] = 0
                judge_scores.append(0)
                per_question.append(row)
                continue

            # ── call the judge ──────────────────────────────────────────
            gold_str = " / ".join(golds)
            score    = call_judge(client, question, gold_str, pred)

            if score is not None:
                judge_scores.append(score)
                row["llm_judge"] = score
            else:
                # API failure: excluded from average, row keeps llm_judge=None
                logger.warning("[%s] judge returned None  id=%r", name, q_id)

        except Exception as exc:
            logger.error(
                "[%s] error processing  id=%r: %s", name, q_id, exc, exc_info=True
            )

        per_question.append(row)

    judged = len(judge_scores)
    pct    = (sum(judge_scores) / judged * 100) if judged > 0 else None

    return {
        "dataset":          name,
        "total":            total,
        "judged":           judged,         # excludes API failures and unmatched items
        "hits1_strict":     None,
        "hits1_flexible":   None,
        "layer_breakdown":  None,
        "llm_judge":        round(pct, 2) if pct is not None else None,
        "per_question":     per_question,
    }


# ── terminal summary ──────────────────────────────────────────────────────

def print_summary(results: list[dict]) -> None:
    W = 52
    print("\n" + "=" * W)
    print(f"  {'Dataset':<10} {'Total':>7} {'Judged':>8} {'Score (%)':>12}")
    print("-" * W)
    for r in results:
        score = f"{r['llm_judge']:.2f}" if r["llm_judge"] is not None else "N/A"
        print(
            f"  {r['dataset']:<10} "
            f"{r['total']:>7} "
            f"{r.get('judged', '?'):>8} "
            f"{score:>12}"
        )
    print("=" * W)


# ── entry point ───────────────────────────────────────────────────────────

def main() -> None:
    if not API_KEY:
        print("ERROR: API_KEY is not set in environment / .env — aborting.")
        return

    client       = OpenAI(api_key=API_KEY)
    all_results: list[dict] = []

    for name, cfg in DATASETS.items():
        print(f"\n── {name.upper()} " + "─" * 44)
        try:
            result = evaluate_dataset(name, cfg, client)
        except FileNotFoundError as exc:
            print(f"  [SKIP] {exc}")
            continue

        out_path = RESULTS_DIR / f"{name}_llm_judge.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        score_str = f"{result['llm_judge']:.2f}%" if result["llm_judge"] is not None else "N/A"
        print(
            f"  Score={score_str}  "
            f"({result.get('judged','?')}/{result['total']} judged)  "
            f"→ saved {out_path}"
        )
        all_results.append(result)

    if all_results:
        print_summary(all_results)

    print(f"\nDone.  Errors logged → {RESULTS_DIR / 'llm_judge_errors.log'}")


if __name__ == "__main__":
    main()
