
import json, re, unicodedata, os, csv
from typing import Dict, Any, List
from collections import Counter

GENERIC_SUFFIXES = {"language", "state", "system", "world series"}

def normalize(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.lower()
    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = re.sub(r'[^\w\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def tokenize(s: str) -> List[str]:
    return normalize(s).split()

def to_canonical_list(text: str) -> List[str]:
    if text is None:
        return []
    raw = str(text)
    if "," in raw:
        parts = [p.strip() for p in raw.split(",")]
    elif re.search(r'\band\b', raw, flags=re.I):
        parts = [p.strip() for p in re.split(r'\band\b', raw, flags=re.I)]
    else:
        parts = [raw.strip()]
    out, seen = [], set()
    for p in parts:
        n = normalize(p)
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out

def strip_generic_suffixes(text: str) -> str:
    t = normalize(text)
    for suf in GENERIC_SUFFIXES:
        if t.endswith(" " + suf):
            t = t[: -len(suf)-1].strip()
    return t

def f1_score(prediction: str, ground_truth: str) -> float:
    pred_toks = tokenize(prediction)
    gold_toks = tokenize(ground_truth)
    common = Counter(pred_toks) & Counter(gold_toks)
    num_same = sum(common.values())
    if len(pred_toks) == 0 or len(gold_toks) == 0:
        return float(pred_toks == gold_toks)
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)

def exact_match(prediction: str, ground_truth: str) -> int:
    return int(normalize(prediction) == normalize(ground_truth))

def year_only(s: str) -> str:
    import re
    m = re.search(r'\b(18|19|20)\d{2}\b', s or "")
    return m.group(0) if m else ""

def is_no_answer(pred: str) -> bool:
    p = normalize(pred)
    return p in {"", "no answer", "none", "n a", "unknown", "not found", "noanswer", "null"}

def load_predictions(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    mapping = {}
    for ex in data:
        pid = ex.get("id") or ex.get("ID") or ex.get("Id")
        ans_obj = ex.get("answer", {})
        pred_text = None
        try:
            pred_text = ans_obj.get("answer", {}).get("final_answer")
        except AttributeError:
            pred_text = None
        if not pred_text:
            a = ans_obj.get("answer")
            if isinstance(a, (str, int, float)):
                pred_text = str(a)
            elif isinstance(a, dict) and "answer" in a and isinstance(a["answer"], str):
                pred_text = a["answer"]
        if pred_text is None:
            pred_text = ""
        mapping[str(pid)] = str(pred_text)
    return mapping

def load_dataset(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    mapping = {}
    for ex in data:
        did = ex.get("ID") or ex.get("id")
        gold = ex.get("answer", "")
        mapping[str(did)] = str(gold)
    return mapping

def list_overlap_metrics(pred: str, gold: str):
    pred_list = to_canonical_list(pred)
    gold_list = to_canonical_list(gold)
    if len(pred_list) >= 2 or len(gold_list) >= 2:
        pred_set = set(pred_list)
        gold_set = set(gold_list)
        tp = len(pred_set & gold_set)
        prec = tp / max(1, len(pred_set))
        rec = tp / max(1, len(gold_set))
        f1 = 0.0 if (prec + rec) == 0 else 2*prec*rec/(prec+rec)
        return tp, prec, rec, f1, pred_list, gold_list
    return 0, 0.0, 0.0, 0.0, pred_list, gold_list

def contains_gold_item(pred: str, gold: str) -> bool:
    pred_list = to_canonical_list(pred)
    gold_list = to_canonical_list(gold)
    if len(gold_list) == 1 and len(pred_list) >= 1:
        return gold_list[0] in set(pred_list) or strip_generic_suffixes(gold_list[0]) in set(pred_list)
    if len(pred_list) == 1 and len(gold_list) >= 1:
        return pred_list[0] in set(gold_list) or pred_list[0] == strip_generic_suffixes(gold_list[0])
    return False

def classify(pred: str, gold: str):
    if is_no_answer(pred):
        return "NO_ANSWER", "prediction marked as no-answer", {}
    em = exact_match(pred, gold)
    f1 = f1_score(pred, gold)
    ypred, ygold = year_only(pred), year_only(gold)
    year_em = (ypred != "" and ygold != "" and ypred == ygold)
    tp, lprec, lrec, lf1, pred_list, gold_list = list_overlap_metrics(pred, gold)
    cg = contains_gold_item(pred, gold)

    if em or year_em or f1 >= 0.85 or cg:
        return "CORRECT", f"em={em}, year_em={year_em}, f1={round(f1,3)}, contains_gold={cg}", {
            "EM": em, "F1": round(f1,3), "YearEM": year_em, "ListTP": tp, "ListF1": round(lf1,3)
        }
    if f1 >= 0.5 or lf1 >= 0.5 or (tp > 0):
        return "PARTIAL", f"f1={round(f1,3)}, list_f1={round(lf1,3)}, list_tp={tp}", {
            "EM": em, "F1": round(f1,3), "YearEM": year_em, "ListTP": tp, "ListF1": round(lf1,3)
        }
    return "WRONG", f"low overlap: f1={round(f1,3)}, list_f1={round(lf1,3)}", {
        "EM": em, "F1": round(f1,3), "YearEM": year_em, "ListTP": tp, "ListF1": round(lf1,3)
    }

def evaluate(pred_path: str, gold_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    preds = load_predictions(pred_path)
    golds = load_dataset(gold_path)

    ids = sorted(set(golds.keys()) & set(preds.keys()))
    missing_pred = sorted(set(golds.keys()) - set(preds.keys()))
    missing_gold = sorted(set(preds.keys()) - set(golds.keys()))

    counters = Counter()
    detailed = []

    for i in ids:
        p = preds[i]
        g = golds[i]
        label, reason, feats = classify(p, g)
        counters[label] += 1
        detailed.append({
            "id": i,
            "label": label,
            "reason": reason,
            "prediction": p,
            "gold": g,
            **feats
        })

    totals = sum(counters.values()) or 1
    metrics = {
        "num_examples": totals,
        "counts": dict(counters),
        "ratios": {k: v / totals for k, v in counters.items()},
        "missing_predictions": len(missing_pred),
        "pred_without_gold": len(missing_gold)
    }

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    with open(os.path.join(out_dir, "detailed.tsv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["id","label","reason","EM","F1","YearEM","ListTP","ListF1","prediction","gold"])
        for r in detailed:
            w.writerow([r["id"], r["label"], r["reason"], r.get("EM",""), r.get("F1",""), r.get("YearEM",""),
                        r.get("ListTP",""), r.get("ListF1",""), r["prediction"], r["gold"]])

    with open(os.path.join(out_dir, "missing_pred_ids.txt"), "w", encoding="utf-8") as f:
        for mid in missing_pred:
            f.write(mid + "\n")
    with open(os.path.join(out_dir, "pred_without_gold_ids.txt"), "w", encoding="utf-8") as f:
        for mid in missing_gold:
            f.write(mid + "\n")

    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--out", default="cwq_eval_plus_run")
    args = parser.parse_args()
    evaluate(args.pred, args.gold, args.out)

# python evaluate_cwq.py --pred ToG_cwq.json --gold ./data/cwq.json --out runs/cwq