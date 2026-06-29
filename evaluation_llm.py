import os
import json
import time
from typing import List, Dict, Any
import openai
from pathlib import Path

# CONFIG
MODEL_NAME = "gpt-4o-mini"  # per your request
BATCH_SIZE = 6
MAX_RETRIES = 3
SLEEP_BETWEEN_RETRIES = 1.5
RESULTS_DIR = Path("runs/llm")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError("Set API_KEY environment variable before running.")
openai.api_key = API_KEY

# Helper: normalize input item to have candidate_answer
def extract_candidate_answer(item: Dict[str, Any]) -> str:
    # format A: answer.answer.final_answer
    try:
        a = item.get("answer", {})
        nested = a.get("answer")
        if isinstance(nested, dict):
            # format A
            final_ans = nested.get("final_answer")
            if final_ans is not None:
                return str(final_ans)
        # format B possibility: answer.answer is a string OR answer is string
        if isinstance(nested, str):
            return nested
        # fallback 1: answer might be directly a string
        if isinstance(a, str):
            return a
        # fallback 2: maybe answer.answer contains explanation + final
        if isinstance(nested, dict):
            return nested.get("final_answer", nested.get("explanation", ""))
    except Exception:
        pass
    return ""  # empty answer if not found

# Transform a list of normalized items into the prompt-required array
def build_batch_payload(items: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    payload = []
    for it in items:
        payload.append({
            "id": it["id"],
            "question": it.get("question", ""),
            "candidate_answer": extract_candidate_answer(it)
        })
    return payload

# The system and user prompt (we will include the items in the user prompt)
SYSTEM_PROMPT = """You are an evaluation assistant. For each input object you must return a single valid JSON object (or an array when multiple inputs are provided). Do NOT include any explanatory text outside the JSON. Follow the schema:

{
  "id": "<input id>",
  "classification": "wrong" | "partially_correct" | "correct",
  "reason": "<one short sentence (8-25 words)>",
  "confidence": <float between 0.0 and 1.0> // optional
}

Classification guidance:
- "correct": answer correctly and fully addresses the question.
- "partially_correct": answer addresses the question partially or contains minor factual errors.
- "wrong": answer is not correct or fails to address the question.

Return concise reasons that explicitly link to the question or to the factual error. If an answer is ambiguous, prefer "partially_correct" with a reason.

Produce only valid JSON.
"""

def build_user_prompt(batch_payload: List[Dict[str, str]]) -> str:
    return (
        "Input items: " + json.dumps(batch_payload, ensure_ascii=False, indent=2)
        + "\nFor each item evaluate whether candidate_answer is correct, partially_correct, or wrong.\n"
        "Return a JSON array of objects following the schema exactly (id, classification, reason, confidence).\n"
        "Reason must be one sentence between 8 and 25 words. Only return the JSON array.\n"
    )

# Parse and validate a single item from model output
def validate_and_normalize(obj: Dict[str, Any]) -> Dict[str, Any]:
    # basic checks
    if "id" not in obj or "classification" not in obj or "reason" not in obj:
        raise ValueError("Missing required fields in model output")
    cls = obj["classification"]
    if cls not in ("wrong", "partially_correct", "correct"):
        # try to map common synonyms
        mapping = {
            "incorrect": "wrong",
            "partial": "partially_correct",
            "partial_correct": "partially_correct",
            "partially correct": "partially_correct",
            "right": "correct",
            "true": "correct",
            "false": "wrong",
        }
        cls_norm = mapping.get(str(cls).lower())
        if cls_norm is None:
            raise ValueError(f"Unknown classification '{cls}'")
        else:
            cls = cls_norm
    reason = " ".join(str(obj["reason"]).split())[:300]  # normalize spacing and cap length
    confidence = obj.get("confidence", None)
    if confidence is not None:
        try:
            confidence = float(confidence)
            confidence = max(0.0, min(1.0, confidence))
        except Exception:
            confidence = None
    return {
        "id": obj["id"],
        "classification": cls,
        "reason": reason,
        "confidence": confidence
    }

# Send one batch to LLM and get parsed array of results
def evaluate_batch_with_retries(batch_payload: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    user_prompt = build_user_prompt(batch_payload)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = openai.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=1200
            )

            text = resp.choices[0].message.content.strip()
            # Try to parse as JSON. Allow models that return a top-level object or array
            parsed = None
            # Some models mistakenly output newlines; try to find the first JSON substring
            def extract_json_substring(s: str):
                s = s.strip()
                # try direct parse
                try:
                    return json.loads(s)
                except Exception:
                    pass
                # try to find first '[' and last ']' pair
                if "[" in s and "]" in s:
                    start = s.find("[")
                    end = s.rfind("]") + 1
                    try:
                        return json.loads(s[start:end])
                    except Exception:
                        pass
                # try to find first '{' and last '}' (single object)
                if "{" in s and "}" in s:
                    start = s.find("{")
                    end = s.rfind("}") + 1
                    try:
                        return json.loads(s[start:end])
                    except Exception:
                        pass
                raise ValueError("Could not extract JSON from model response")
            parsed = extract_json_substring(text)
            # parsed should be an array
            if isinstance(parsed, dict):
                parsed = [parsed]
            if not isinstance(parsed, list):
                raise ValueError("Model JSON parsed to non-list")
            # validate each item
            normalized = [validate_and_normalize(p) for p in parsed]
            # basic sanity check: ids returned should match input ids (or at least be subset)
            input_ids = set([it["id"] for it in batch_payload])
            parsed_ids = set([p["id"] for p in normalized])
            if not parsed_ids.issubset(input_ids):
                # still accept, but warn (we will let the code continue)
                print("Warning: model returned ids not in input; continuing with matching ids.")
            # return those that match inputs (maintain input order)
            id2 = {p["id"]: p for p in normalized}
            results_in_order = [id2[i["id"]] for i in batch_payload if i["id"] in id2]
            return results_in_order
        except Exception as e:
            print(f"[Attempt {attempt}] Error parsing model response: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(SLEEP_BETWEEN_RETRIES * attempt)
                continue
            else:
                raise
    # unreachable
    return []

# High-level evaluate function for the whole file
def evaluate_file(input_file: str):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)[:10]
    if not isinstance(data, list):
        raise ValueError("Input JSON file must contain a list of objects at top level.")

    items_to_evaluate = []  # items to send to LLM
    combined_results = []

    for obj in data:
        candidate = extract_candidate_answer(obj)
        item_id = obj.get("id")
        question = obj.get("question", "")

        # Check if answer is "No Answer" (or empty string)
        if str(candidate).strip().lower() == "no answer" or candidate.strip() == "":
            res = {
                "id": item_id,
                "classification": "unknown",
                "reason": "No answer provided in the original data.",
                "confidence": None
            }
            combined_results.append(res)
            # write per-id file immediately
            out_path = RESULTS_DIR / f"{item_id}.json"
            with open(out_path, "w", encoding="utf-8") as wf:
                json.dump(res, wf, ensure_ascii=False, indent=2)
        else:
            # keep items for LLM evaluation
            items_to_evaluate.append({
                "id": item_id,
                "question": question,
                "candidate_answer": candidate,
                "orig": obj
            })

    # Batch and evaluate items with valid answers
    for i in range(0, len(items_to_evaluate), BATCH_SIZE):
        batch = items_to_evaluate[i:i+BATCH_SIZE]
        batch_payload = [{"id": it["id"], "question": it["question"], "candidate_answer": it["candidate_answer"]} for it in batch]
        try:
            batch_results = evaluate_batch_with_retries(batch_payload)
        except Exception as e:
            print(f"Failed to evaluate batch starting at index {i}: {e}")
            batch_results = []
            for it in batch:
                batch_results.append({
                    "id": it["id"],
                    "classification": "wrong",
                    "reason": "Evaluation failed due to API/parse error.",
                    "confidence": None
                })
        # write per-id files
        for res in batch_results:
            out_path = RESULTS_DIR / f"{res['id']}.json"
            with open(out_path, "w", encoding="utf-8") as wf:
                json.dump(res, wf, ensure_ascii=False, indent=2)
        combined_results.extend(batch_results)

    # also write combined file as JSON Lines
    combined_path = RESULTS_DIR / "combined.jsonl"
    with open(combined_path, "w", encoding="utf-8") as cf:
        for r in combined_results:
            cf.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Done. Results written to {RESULTS_DIR.resolve()}")
