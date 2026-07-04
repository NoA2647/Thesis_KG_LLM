#!/usr/bin/env python3
"""
hit1-evaluate.py
Evaluates KGQA system output with Hits@1 (strict + flexible, 4 layers).
Fully independent from llm-as-judge-evaluate.py.

Gold files   : data/cwq.json, data/qald.json, data/webq.json
Output files : data/cwq_output.json, data/qald_output.json, data/webq_output.json
Results      : eval_results/{dataset}_hits1.json
Error log    : eval_results/hit1_errors.log
"""

import json
import re
import string
import logging
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# ── paths & dataset config ───────────────────────────────────────────────
DATA_DIR    = Path("data")
RESULTS_DIR = Path("eval_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)   # must exist before logging setup

DATASETS: dict[str, dict] = {
    "cwq": {
        "gold":       DATA_DIR / "cwq.json",
        "output":     DATA_DIR / "cwq_output.json",
        "id_key":     "ID",       # gold item key used as match key
        "answer_key": "answer",   # single string
        "match_by":   "id",       # match output→gold by id field
    },
    "qald": {
        "gold":       DATA_DIR / "qald_10-en.json",
        "output":     DATA_DIR / "qald_output.json",
        "id_key":     None,
        "answer_key": "answer",   # dict {qid: label} → use values
        "match_by":   "question", # match by exact question string
    },
    "webq": {
        "gold":       DATA_DIR / "webquestions.json",
        "output":     DATA_DIR / "webquestions_output.json",
        "id_key":     None,
        "answer_key": "answers",  # list of strings
        "match_by":   "question",
    },
}

# strings that mean "system has no answer" — count as wrong, skip LLM
NO_ANSWER_STRINGS: frozenset[str] = frozenset({"no answer", "none", "n/a", ""})

# ── logging ──────────────────────────────────────────────────────────────
logging.basicConfig(
    filename=str(RESULTS_DIR / "hit1_errors.log"),
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)


# ── text normalization ───────────────────────────────────────────────────

def normalize(text: str) -> str:
    """Lowercase → strip punctuation → collapse whitespace → drop leading article."""
    if not text:
        return ""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    if tokens and tokens[0] in ("the", "a", "an"):
        tokens = tokens[1:]
    return " ".join(tokens)


# ── 4-layer matching ─────────────────────────────────────────────────────

def match_layer(pred: str, gold: str) -> Optional[int]:
    """
    Try layers 1-4 in order; return the first that passes, or None.
      L1 - exact match of normalized strings
      L2 - token-set match (same tokens, any order)
      L3 - containment  (all gold tokens ⊆ pred tokens)
      L4 - token F1 >= 0.8
    """
    np = normalize(pred)
    ng = normalize(gold)
    np_toks = set(np.split())
    ng_toks = set(ng.split())

    # L1
    if np == ng:
        return 1
    # L2
    if np_toks and ng_toks and np_toks == ng_toks:
        return 2
    # L3
    if ng_toks and ng_toks.issubset(np_toks):
        return 3
    # L4
    if np_toks and ng_toks:
        common   = np_toks & ng_toks
        prec     = len(common) / len(np_toks)
        rec      = len(common) / len(ng_toks)
        denom    = prec + rec
        if denom > 0 and (2 * prec * rec / denom) >= 0.8:
            return 4
    return None


def evaluate_item(
    pred: Optional[str],
    gold_list: list[str],
) -> tuple[bool, bool, Optional[int]]:
    """
    Returns (strict, flexible, best_matched_layer).
    strict    = any gold passes L1
    flexible  = any gold passes any layer (L1-L4)
    best_layer = earliest layer that matched across all golds, or None
    """
    if not pred or normalize(pred) in NO_ANSWER_STRINGS:
        return False, False, None

    strict    = any(normalize(pred) == normalize(g) for g in gold_list)
    best_layer: Optional[int] = None
    for g in gold_list:
        layer = match_layer(pred, g)
        if layer is not None and (best_layer is None or layer < best_layer):
            best_layer = layer

    return strict, (best_layer is not None), best_layer


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


# ── per-dataset evaluation ───────────────────────────────────────────────

def evaluate_dataset(name: str, cfg: dict) -> dict:
    with open(cfg["gold"],   encoding="utf-8") as f:
        gold_data: list[dict] = json.load(f)
    with open(cfg["output"], encoding="utf-8") as f:
        output_data: list[dict] = json.load(f)

    # build lookup: key → gold item
    if cfg["match_by"] == "id":
        gold_lookup = {str(item[cfg["id_key"]]): item for item in gold_data}
    else:
        # exact question string — no normalization (as per spec)
        gold_lookup = {item["question"]: item for item in gold_data}

    per_question: list[dict] = []
    layer_breakdown: dict[str, int] = {"L1": 0, "L2": 0, "L3": 0, "L4": 0, "none": 0}
    strict_total = flexible_total = total = 0

    for item in tqdm(output_data, desc=f"  hits@1 / {name}", unit="q", ncols=82):
        total += 1
        q_id     = str(item.get("id", ""))
        question = item.get("question", "")

        # default row (wrong at all layers)
        row: dict = {
            "id":             q_id,
            "question":       question,
            "predicted":      None,
            "gold":           [],
            "hits1_strict":   False,
            "hits1_flexible": False,
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
                layer_breakdown["none"] += 1
                per_question.append(row)
                continue

            pred  = extract_predicted(item)
            golds = extract_gold_list(gold_item, cfg["answer_key"])

            strict, flexible, best_layer = evaluate_item(pred, golds)

            strict_total   += int(strict)
            flexible_total += int(flexible)
            lk = f"L{best_layer}" if best_layer else "none"
            layer_breakdown[lk] += 1

            row.update(
                predicted=pred, gold=golds,
                hits1_strict=strict, hits1_flexible=flexible,
                matched_layer=best_layer,
            )

        except Exception as exc:
            logger.error(
                "[%s] error processing  id=%r: %s", name, q_id, exc, exc_info=True
            )
            layer_breakdown["none"] += 1

        per_question.append(row)

    hits1_strict   = strict_total   / total if total else 0.0
    hits1_flexible = flexible_total / total if total else 0.0

    return {
        "dataset":          name,
        "total":            total,
        "hits1_strict":     round(hits1_strict,   4),
        "hits1_flexible":   round(hits1_flexible, 4),
        "layer_breakdown":  layer_breakdown,
        "llm_judge":        None,
        "per_question":     per_question,
    }


# ── terminal summary ─────────────────────────────────────────────────────

def print_summary(results: list[dict]) -> None:
    W = 70
    print("\n" + "=" * W)
    print(f"  {'Dataset':<10} {'Strict':>8} {'Flexible':>10}   Layer breakdown")
    print("-" * W)
    for r in results:
        lb = r["layer_breakdown"]
        print(
            f"  {r['dataset']:<10} "
            f"{r['hits1_strict']:>8.4f} "
            f"{r['hits1_flexible']:>10.4f}   "
            f"L1={lb['L1']}  L2={lb['L2']}  "
            f"L3={lb['L3']}  L4={lb['L4']}  none={lb['none']}"
        )
    print("=" * W)


# ── entry point ──────────────────────────────────────────────────────────

def main() -> None:
    all_results: list[dict] = []

    for name, cfg in DATASETS.items():
        print(f"\n── {name.upper()} " + "─" * 44)
        try:
            result = evaluate_dataset(name, cfg)
        except FileNotFoundError as exc:
            print(f"  [SKIP] {exc}")
            continue

        out_path = RESULTS_DIR / f"{name}_hits1.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(
            f"  Strict={result['hits1_strict']:.4f}  "
            f"Flexible={result['hits1_flexible']:.4f}  "
            f"→ saved {out_path}"
        )
        all_results.append(result)

    if all_results:
        print_summary(all_results)

    print(f"\nDone.  Errors logged → {RESULTS_DIR / 'hit1_errors.log'}")


if __name__ == "__main__":
    main()
