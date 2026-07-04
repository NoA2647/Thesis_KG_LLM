import json
import re
import statistics

# ---------------------------
# Configuration for relaxed matching
# ---------------------------
IGNORE_WORDS = ["language", "dialect", "people", "official", "spoken", "the", "of", "and"]

def normalize(text):
    """Lowercase, remove punctuation, and ignore common words."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    for w in IGNORE_WORDS:
        text = re.sub(r"\b" + re.escape(w) + r"\b", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def is_relaxed_match(pred, truths):
    """Return True if any truth matches prediction in a relaxed way."""
    pred_norm = normalize(pred)
    for truth in truths:
        truth_norm = normalize(truth)
        if truth_norm in pred_norm or pred_norm in truth_norm:
            return True
    return False

# ---------------------------
# Load predictions and dataset
# ---------------------------
with open("data/webquestions_output.json", "r", encoding="utf-8") as f:
    predictions = json.load(f)

with open("./data/WebQuestions.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

# Map dataset by question for easy lookup
dataset_map = {item["question"]: item["answers"] for item in dataset}

# ---------------------------
# Step 1: Evaluate predictions
# ---------------------------
results = []
correct_count = 0

for pred in predictions:
    question = pred["question"]
    pred_answer = pred["answer"]["answer"]["final_answer"]
    true_answers = dataset_map.get(question, [])
    
    correct = is_relaxed_match(pred_answer, true_answers)
    if correct:
        correct_count += 1
    
    results.append({
        "question": question,
        "prediction": pred_answer,
        "true_answers": true_answers,
        "correct": correct
    })

total_questions = len(predictions)
accuracy = correct_count / total_questions * 100

print(f"\n--- Step 1: Prediction Evaluation ---")
print(f"Total questions: {total_questions}")
print(f"Correct predictions: {correct_count}")
print(f"Accuracy: {accuracy:.2f}%")

# Save detailed results
with open("evaluation_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# ---------------------------
# Step 2: Subquestions analysis
# ---------------------------
subq_stats = []
all_subq_counts = []

for pred in predictions:
    question = pred["question"]
    paths = pred.get("paths", [])
    
    path_subq_counts = [len(path.get("subqs", [])) for path in paths]
    
    if path_subq_counts:
        max_subq = max(path_subq_counts)
        min_subq = min(path_subq_counts)
        avg_subq = sum(path_subq_counts) / len(path_subq_counts)
    else:
        max_subq = min_subq = avg_subq = 0
    
    subq_stats.append({
        "question": question,
        "path_subq_counts": path_subq_counts,
        "max_subq": max_subq,
        "min_subq": min_subq,
        "avg_subq": avg_subq
    })
    
    all_subq_counts.extend(path_subq_counts)

# Overall subquestion statistics
if all_subq_counts:
    overall_avg_subq = sum(all_subq_counts) / len(all_subq_counts)
    overall_median_subq = statistics.median(all_subq_counts)
    overall_min_subq = min(all_subq_counts)
    overall_max_subq = max(all_subq_counts)
else:
    overall_avg_subq = overall_median_subq = overall_min_subq = overall_max_subq = 0

print(f"\n--- Step 2: Subquestions Analysis ---")
print(f"Average subquestions per path: {overall_avg_subq:.2f}")
print(f"Median subquestions per path: {overall_median_subq}")
print(f"Min subquestions per path: {overall_min_subq}")
print(f"Max subquestions per path: {overall_max_subq}")

with open("subquestions_stats.json", "w", encoding="utf-8") as f:
    json.dump(subq_stats, f, indent=2, ensure_ascii=False)

# ---------------------------
# Step 3: Estimated time analysis
# ---------------------------
llm_times = []
wiki_times = []
total_times = []

for pred in predictions:
    est = pred.get("estimate_time", {})
    llm_times.append(est.get("est_llm", 0))
    wiki_times.append(est.get("est_wiki", 0))
    total_times.append(est.get("est_total", 0))

def compute_stats(times_list, label):
    if times_list:
        avg = sum(times_list) / len(times_list)
        med = statistics.median(times_list)
        min_time = min(times_list)
        max_time = max(times_list)
    else:
        avg = med = min_time = max_time = 0
    print(f"{label} Times (s): Avg={avg:.2f}, Median={med:.2f}, Min={min_time:.2f}, Max={max_time:.2f}")

print(f"\n--- Step 3: Estimated Time Analysis ---")
compute_stats(llm_times, "LLM")
compute_stats(wiki_times, "Wiki")
compute_stats(total_times, "Total")

per_question_times = [
    {
        "question": pred["question"],
        "est_llm": pred.get("estimate_time", {}).get("est_llm", 0),
        "est_wiki": pred.get("estimate_time", {}).get("est_wiki", 0),
        "est_total": pred.get("estimate_time", {}).get("est_total", 0)
    }
    for pred in predictions
]

with open("estimated_times.json", "w", encoding="utf-8") as f:
    json.dump(per_question_times, f, indent=2, ensure_ascii=False)

print("\nAll results saved to JSON files: evaluation_results.json, subquestions_stats.json, estimated_times.json")
