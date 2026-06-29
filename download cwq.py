#!/usr/bin/env python3
import requests
import json
import time
from pathlib import Path

OUT = Path("complex_web_questions_test.jsonl")
DATASET = "drt/complex_web_questions"
CONFIG = "complexwebquestions_test"
SPLIT = "test"
LENGTH = 100   # try 200-1000; server may cap it
BASE_URL = "https://datasets-server.huggingface.co/rows"

MAX_RETRIES = 5
BACKOFF_BASE = 0.5

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", encoding="utf-8") as fout:
    offset = 0
    total_fetched = offset

    while True:
        params = {
            "dataset": DATASET,
            "config": CONFIG,
            "split": SPLIT,
            "offset": offset,
            "length": LENGTH,
        }

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = requests.get(BASE_URL, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as e:
                wait = BACKOFF_BASE * (2 ** (attempt - 1))
                print(f"Request error (attempt {attempt}/{MAX_RETRIES}): {e}. Retrying in {wait:.1f}s...")
                time.sleep(wait)
        else:
            raise RuntimeError("Exceeded retries, aborting.")

        # The HF rows endpoint returns an object with 'rows' list
        rows = data.get("rows")
        if rows is None:
            # Fallback: maybe top-level array returned
            if isinstance(data, list):
                rows = data
            else:
                raise RuntimeError("Unexpected response format: 'rows' field not found")

        if not rows:
            print("No more rows. Exiting.")
            break

        for row in rows:
            # Optionally, the 'row' object may have a 'row' field or 'value' — write as-is
            # Prefer writing the row's 'value' if present (common pattern)
            to_write = row.get("row") or row.get("value") or row
            fout.write(json.dumps(to_write, ensure_ascii=False) + "\n")

        fetched = len(rows)
        total_fetched += fetched
        print(f"Fetched {fetched} rows (offset {offset}). Total so far: {total_fetched}")

        # If the server returns a 'total' field, we can stop when offset >= total
        total_available = data.get("total")
        if isinstance(total_available, int) and offset + fetched >= total_available:
            print("All rows fetched (based on total).")
            break

        if fetched < LENGTH:
            print("Last page returned fewer rows than page size — done.")
            break

        offset += fetched
        time.sleep(0.2)  # be polite

print(f"Saved {total_fetched} rows to {OUT.resolve()}")
