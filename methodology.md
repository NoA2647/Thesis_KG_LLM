# Methodology: LLM-Guided Beam Search over a Knowledge Graph for Multi-Hop Question Answering

---

## 1. Overview

The system addresses the task of **multi-hop question answering over a knowledge graph (KG)**. Given a natural-language question and one or more seed entities (topic entities) identified in the question, the goal is to traverse a structured knowledge graph — concretely, Wikidata — through a sequence of relational hops, collect a set of relevant fact chains (paths), and synthesize a final answer using a large language model (LLM).

The system is not a simple retrieval pipeline. It combines three distinct reasoning capabilities:

1. **Decomposition**: an LLM breaks the question into a sequence of targeted sub-questions that guide traversal.
2. **Structured search**: a beam search algorithm explores and prunes paths through the knowledge graph at each hop.
3. **Neural reasoning**: a second LLM call synthesizes the collected evidence paths into a final answer.

A fallback Chain-of-Thought (CoT) path exists for questions that lack identifiable topic entities, in which case the LLM reasons entirely from its parametric knowledge.

---

## 2. Problem Formulation

Given:
- A natural-language question $Q$
- A set of initial topic entities $\mathcal{E}_0 = \{e_1, e_2, \ldots\}$, each identified by a Wikidata QID and a human-readable label

The system produces:
- A final answer string
- A collection of supporting KG paths (chains of subject–predicate–object triples) that justify the answer

Each path is rooted at one initial topic entity. Multiple paths (one per topic entity) are searched independently and their evidence is merged at the reasoning stage.

---

## 3. Data Representation

Before describing the algorithm, it is important to understand the core data structures used throughout.

**TopicEntity** represents any node in the knowledge graph — either an identified Wikidata item (with a QID such as `Q42`) or a literal value (a date, quantity, or string with no QID). Each entity carries a label (human-readable name) and optional qualifier metadata from its Wikidata statements (e.g., temporal qualifiers like "start time" or "end time" attached to a claim).

**RelationEdge** represents a Wikidata property connecting two entities, carrying both the property ID (PID, e.g., `P31`) and its human-readable label (e.g., "instance of").

**Hop** is a single traversed edge in the graph: a (source entity, relation, target entity) triple enriched with a scalar hop score assigned by the LLM pruner.

**Branch** is a walk (path) through the KG accumulated over multiple hops. It stores the ordered list of hops taken so far (history), the current frontier entity at the tip of the walk, and a cumulative score that aggregates all hop scores along the path. A branch is the fundamental unit of state in the beam search.

**SubQuestion** is one decomposed reasoning step produced by the question-decomposition module. It carries a question string (the actionable step to take on the KG) and a purpose string (a one-sentence explanation of why this step is needed).

**PathResult** is the output of searching one initial entity. It stores the initial entity, the list of sub-questions used to guide the search, and the final beam of branches after traversal completes.

---

## 4. Knowledge Graph Access Layer

The system accesses Wikidata through a dedicated client (`WikiDataC_v2`) that wraps the Wikidata MediaWiki API. This client performs three main operations:

**Entity info retrieval**: Given a QID, the client fetches the entity's label, description, and all claims (property–value pairs). Claims are fetched using the `wbgetentities` API endpoint. Because Wikidata returns raw property IDs and value QIDs, the client performs a second batch call to resolve all referenced IDs to their human-readable labels in a single round trip (up to 50 IDs per call). Qualifier metadata attached to each statement (e.g., "point in time", "applies to part") is also resolved and stored alongside the main claim value. This rich qualifier data is later passed to the LLM pruner as `meta_data` on each candidate entity, helping the LLM make temporally or contextually aware pruning decisions.

**Caching**: All resolved entity info is cached locally as a JSON file (keyed by QID). Subsequent accesses are served from disk, avoiding redundant API calls. This dramatically speeds up repeated runs over the same dataset.

**Relation and tail retrieval**: The client exposes two primary query methods used during traversal:
- `get_all_relations_of_an_entity(qid)` — returns all outgoing property IDs and labels for an entity.
- `get_tail_entities_given_head_and_relation(qid, pid)` — returns all tail entities (with labels, QIDs, and qualifier metadata) for a given head–property pair.

Literal values (dates, quantities, strings) are handled gracefully: they receive `qid=None` and are flagged as `is_value=True`, indicating the traversal cannot extend further from them.

---

## 5. Relation Filtering

Before passing relations to the LLM, a lightweight rule-based filter removes obviously noisy or uninformative properties. Relations are discarded if they end with administrative suffixes (e.g., "ID", "code", "URL", "website", "image") or if they correspond to Wikidata-internal housekeeping properties (e.g., "Commons category", "Wikidata usage instructions"). This filter is configurable and reduces the size of the candidate set fed to the LLM, lowering both cost and noise.

---

## 6. Question Decomposition

For each initial topic entity, the system generates a sequence of sub-questions by prompting an LLM. The prompt provides the full main question and the label of the current topic entity. The LLM is instructed to produce a JSON array of steps, each step containing a `sub_question` (an actionable traversal directive starting from the current node) and a `purpose` (a one-sentence justification).

The decomposition is designed to mirror the multi-hop reasoning structure of the question. For example, a question like "What is the predominant religion of the institution led by Ovadia Yosef?" would be decomposed into: (1) identify the religious institution or community led by Ovadia Yosef; (2) find the predominant religion of that institution. Each sub-question corresponds to one hop (or conceptual step) in the knowledge graph traversal.

Crucially, the decomposition is per-initial-entity: different topic entities may yield different sub-question sequences, enabling independent reasoning paths. If decomposition fails (due to an LLM error or parsing failure), the system skips that path and records an empty result.

---

## 7. Beam Search Algorithm

The core traversal algorithm is a **guided beam search** over the knowledge graph. It is "guided" because at every expansion step, an LLM evaluates and scores the available (relation, entity) pairs rather than using a fixed heuristic.

### 7.1 Initialization

The beam is initialized with a single branch: the frontier is set to the initial topic entity, the history is empty, and the cumulative score is zero. The beam width $K$ (default: 3) is fixed for all steps.

### 7.2 Iterative Expansion

The algorithm iterates over the list of sub-questions in order. At each step:

**Expansion phase**: Every branch currently in the beam is expanded. Expansion proceeds as follows:
1. The KG client retrieves all outgoing (relation, candidate entity) pairs from the branch's frontier entity. If the frontier is a literal (no QID), expansion returns nothing and the branch produces no children.
2. The LLM pruner receives the full set of candidate pairs and scores them.
3. Each surviving scored pair becomes a new branch: a copy of the parent branch with the new hop appended to the history, the target entity set as the new frontier, and the cumulative score updated.

The expansion of all branches in the beam produces a pool of candidate branches for the next step.

**Selection phase**: The top-$K$ branches from the pool are selected using a deterministic sort: primary key is cumulative score (descending), with tie-breaking on the frontier entity's QID and relation PID (ascending). This ensures reproducibility. The selected $K$ branches become the beam for the next sub-question step.

**Early termination**: If the pool of expanded branches is empty at any step (e.g., because all frontier entities are literals or the KG returned no useful neighbors), the search terminates early and returns the last non-empty beam as the final result.

### 7.3 LLM Pruning and Scoring

The pruner is the most nuanced component of the beam search. At each expansion step, the LLM receives a detailed context prompt containing:
- The main question
- The initial (root) node of this path
- All sub-questions already traversed in this reasoning path
- All nodes already visited (to prevent cycles)
- The current node being expanded
- The current sub-question and its purpose
- A numbered list of all available (relation, entity) pairs, each annotated with the entity's qualifier metadata

The LLM is instructed to select exactly $K$ pairs most relevant to the current sub-question and to assign each a probability score in $[0, 1]$ that sums to 1.0. The output is a JSON array of objects containing the selected relation label, entity label, a brief justification sentence, and a score.

The scores are then matched back to the full typed pair objects (containing PIDs, QIDs, and metadata) by comparing string labels, producing a list of `Hop` objects ready for the score-combination step.

This approach is powerful because the LLM can read the metadata qualifiers on candidate entities and understand temporal or contextual constraints (e.g., preferring the "head coach" with the most recent "end time" qualifier), and can consider the traversal history to avoid backtracking or selecting redundant nodes.

### 7.4 Score Combination

Each hop carries a scalar score assigned by the LLM (between 0 and 1). The cumulative score of a branch is updated at each hop using one of three configurable combination rules:

- **Add**: $\text{cum}_{t+1} = \text{cum}_t + \text{hop}_t$. Simple sum; later hops weigh equally.
- **Multiply**: $\text{cum}_{t+1} = \text{cum}_t \times \max(\text{hop}_t, \epsilon)$. Product of hop scores; longer paths are penalized if any hop is weak.
- **Log**: $\text{cum}_{t+1} = \text{cum}_t + \log(\max(\text{hop}_t, \epsilon))$. Log-sum; equivalent to the log-probability of the path under an independence assumption.

The combination rule is chosen at configuration time and applies uniformly across all paths.

---

## 8. Multi-Path Orchestration

The pipeline manages one independent beam search per initial topic entity. This means if a question has two topic entities (e.g., both a person and a location), two separate beam searches run, each with their own sub-question decomposition and traversal. The resulting `PathResult` objects from all paths are collected and passed together to the final reasoning stage.

This design allows the system to gather evidence from multiple perspectives and lets the reasoner synthesize information that spans more than one entry point into the knowledge graph.

An optional `max_paths` cap limits the number of initial entities processed, useful for controlling cost when a question has many identified entities.

---

## 9. Final Reasoning

After all path searches complete, a second LLM call synthesizes all collected evidence into a final answer. All branches from all paths are serialized into a structured list of knowledge triplets. Each branch's history is rendered as a chain of head–relation–tail entries, each enriched with the tail entity's metadata. This representation mirrors the format expected by the reasoning prompt.

The reasoning prompt presents the main question and the full set of extracted KG triplets, then asks the LLM to determine:
- Whether the KG information is **sufficient** to answer the question (`true`/`false`)
- A brief **explanation** of why the information is or is not sufficient
- The **final answer** derived from the evidence (or "No Answer" if insufficient)

The reasoning LLM is instructed to use the KG triplets as primary evidence but may also draw on its parametric knowledge if the triplets alone are ambiguous or incomplete. The output is a structured JSON object, which the system parses and returns as the final result.

---

## 10. Fallback: Chain-of-Thought Reasoning

When a question arrives with no identifiable initial topic entities (i.e., the dataset row contains no QID mapping), the system bypasses the KG traversal entirely and falls back to pure LLM reasoning. A Chain-of-Thought prompt presents the question and asks the LLM to reason step by step, producing a JSON object with a reasoning sentence and a final answer. This fallback ensures the system always produces an output even when structured KG grounding is unavailable.

---

## 11. Output and Persistence

For each processed question, the system writes a result record to a JSON file. Each record contains the question's dataset ID, the full question text, the structured answer object (sufficient flag, explanation, final answer), and the serialized path results (all branches with their complete hop histories, scores, and entity metadata). Records are appended atomically using a write-then-replace strategy to prevent file corruption across long runs. This output format supports post-hoc analysis of the traversal paths, visualization of the KG subgraph explored, and comparison of predicted answers against ground truth.

---

## 12. System Configuration

All behavioral parameters are centralized in a configuration dataclass (`RunConfig`) loaded at startup. Key parameters include:

- **Beam width** ($K$): number of branches retained at each search step (default: 3).
- **Combine rule**: score accumulation strategy (`add`, `mul`, or `log`).
- **Remove unnecessary relations**: whether to apply rule-based relation filtering before LLM pruning.
- **LLM type**: the model used for all LLM calls (supports OpenAI-compatible endpoints including local models via Ollama).
- **Temperature (exploration)**: temperature for sub-question generation and LLM pruning calls (default: 0.0 for determinism).
- **Temperature (reasoning)**: temperature for the final reasoning call.
- **Max paths**: optional cap on the number of initial entities processed per question.
- **Dataset**: the benchmark dataset being evaluated (determines file paths and field names for question and ID extraction).

---

## 13. End-to-End Algorithm Summary

The full pipeline for one question proceeds as follows:

1. Load the question and its initial topic entities from the dataset.
2. If no topic entities are available, run CoT fallback and output the result.
3. Otherwise, for each initial topic entity:
   a. Call the LLM to decompose the main question into a sequence of sub-questions relative to this entity.
   b. Initialize a beam with a single branch rooted at this entity.
   c. For each sub-question in order:
      - For each branch in the current beam, fetch all outgoing (relation, entity) pairs from the frontier node via the Wikidata API.
      - Call the LLM pruner to score and select the top-$K$ pairs for each branch.
      - Expand each branch into child branches using the scored pairs.
      - Collect all child branches from all beam members into a pool.
      - Select the top-$K$ branches from the pool by cumulative score.
      - Advance the beam to the next step.
   d. Record the final beam as this path's result.
4. Collect all path results across all initial entities.
5. Serialize all final branches as knowledge triplets and call the LLM reasoner.
6. Parse the structured JSON answer and write the full result to disk.
