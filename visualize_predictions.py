import json
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# ---------------------------
# Load JSON files
# ---------------------------
with open("subquestions_stats.json", "r", encoding="utf-8") as f:
    subq_stats = json.load(f)

with open("estimated_times.json", "r", encoding="utf-8") as f:
    times_stats = json.load(f)

# ---------------------------
# Prepare Data
# ---------------------------
all_subq_counts = []
avg_subq_per_question = []
for item in subq_stats:
    all_subq_counts.extend(item["path_subq_counts"])
    avg_subq_per_question.append(item["avg_subq"])

llm_times = [item["est_llm"] for item in times_stats]
wiki_times = [item["est_wiki"] for item in times_stats]
total_times = [item["est_total"] for item in times_stats]

# ---------------------------
# 1️⃣ Subquestions Histogram
# ---------------------------
plt.figure(figsize=(8,5))
plt.hist(all_subq_counts, bins=range(1, max(all_subq_counts)+2), color='skyblue', edgecolor='black')
plt.title("Distribution of Subquestions per Path")
plt.xlabel("Number of Subquestions")
plt.ylabel("Frequency")
plt.xticks(range(1, max(all_subq_counts)+2))
plt.grid(axis='y', alpha=0.75)
plt.savefig("subquestions_histogram.png")
plt.show()

# ---------------------------
# 2️⃣ Estimated Times Histograms
# ---------------------------
plt.figure(figsize=(10,6))
plt.hist(llm_times, bins=10, alpha=0.6, label="LLM", color="orange", edgecolor='black')
plt.hist(wiki_times, bins=10, alpha=0.6, label="Wiki", color="green", edgecolor='black')
plt.hist(total_times, bins=10, alpha=0.6, label="Total", color="blue", edgecolor='black')
plt.title("Distribution of Estimated Times (seconds)")
plt.xlabel("Estimated Time (s)")
plt.ylabel("Frequency")
plt.legend()
plt.grid(axis='y', alpha=0.75)
plt.savefig("estimated_times_histogram.png")
plt.show()

# ---------------------------
# 3️⃣ Boxplots for Estimated Times
# ---------------------------
plt.figure(figsize=(8,6))
plt.boxplot([llm_times, wiki_times, total_times], labels=["LLM", "Wiki", "Total"])
plt.title("Boxplot of Estimated Times")
plt.ylabel("Estimated Time (s)")
plt.grid(axis='y', alpha=0.75)
plt.savefig("estimated_times_boxplot.png")
plt.show()

# ---------------------------
# 4️⃣ Scatter Plot: Avg Subquestions vs Total Time
# ---------------------------
plt.figure(figsize=(8,6))
plt.scatter(avg_subq_per_question, total_times, color="purple", label="Data points")

# Optional: Add linear trend line using Pearson correlation
if len(avg_subq_per_question) > 1:
    m, b = pearsonr(avg_subq_per_question, total_times)[0], 0
    plt.plot(avg_subq_per_question, [m*x + min(total_times) for x in avg_subq_per_question],
             color="red", linestyle="--", label="Trend line")

plt.title("Average Subquestions vs Total Estimated Time")
plt.xlabel("Average Subquestions per Question")
plt.ylabel("Total Estimated Time (s)")
plt.legend()
plt.grid(alpha=0.5)
plt.savefig("subquestions_vs_total_time_scatter_with_corr.png")
plt.show()

# ---------------------------
# 5️⃣ Correlation Analysis
# ---------------------------
if len(avg_subq_per_question) > 1:
    pearson_corr, pearson_p = pearsonr(avg_subq_per_question, total_times)
    spearman_corr, spearman_p = spearmanr(avg_subq_per_question, total_times)
    print("--- Correlation Analysis ---")
    print(f"Pearson correlation: {pearson_corr:.3f} (p-value={pearson_p:.3f})")
    print(f"Spearman correlation: {spearman_corr:.3f} (p-value={spearman_p:.3f})")
else:
    print("Not enough data to compute correlations.")
