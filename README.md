# Knowledge-Grounded LLM Reasoning for Reliable Multi-Hop Question Answering

## Overview

Large Language Models (LLMs) demonstrate impressive question-answering capabilities but can produce hallucinated or unsupported responses, particularly when answering complex questions requiring multi-hop reasoning.

This project presents a **training-free, knowledge-graph-grounded framework** for improving the reasoning reliability, evidence grounding, and explainability of LLMs in multi-hop question answering.

The framework integrates LLM-based question decomposition, knowledge graph exploration, candidate relation–entity ranking, and evidence-sufficiency assessment to generate answers supported by explicit knowledge graph paths. It also incorporates an abstention mechanism that allows the system to return *No Answer* when the available evidence is insufficient.

This research was conducted as part of my MSc thesis in Computer Science (Artificial Intelligence) at Amirkabir University of Technology, Tehran, Iran.

## Research Objectives

The main objectives of this research are:

- Improve LLM reasoning over complex questions requiring multiple reasoning steps.
- Ground generated answers in structured, verifiable knowledge.
- Reduce unsupported claims and hallucinations through evidence-based reasoning.
- Improve answer explainability by identifying supporting knowledge graph paths.
- Enable reliable abstention when retrieved evidence is insufficient.

## Proposed Framework

The proposed framework combines LLM reasoning with structured knowledge graph exploration.

### Workflow

1. **Question Decomposition:** An LLM decomposes a complex question into smaller, more manageable sub-questions.
2. **Knowledge Graph Exploration:** A beam search algorithm explores Wikidata to identify candidate entities and relations relevant to the question.
3. **Candidate Ranking:** An LLM evaluates and ranks candidate relation–entity pairs at each reasoning hop.
4. **Evidence Collection:** Supporting knowledge graph paths are collected to provide structured evidence for the reasoning process.
5. **Evidence-Based Answer Generation:** A reasoning LLM generates the final answer using the retrieved evidence.
6. **Evidence-Sufficiency Assessment:** The system assesses whether the available evidence sufficiently supports the generated answer.
7. **Abstention:** When the evidence is insufficient, the framework can return a *No Answer* response rather than generate an unsupported answer.

## Key Contributions

- A training-free framework for knowledge-grounded multi-hop question answering.
- Integration of LLM-based question decomposition and knowledge graph search.
- Beam-search-based exploration of candidate reasoning paths.
- LLM-guided ranking of candidate relations and entities.
- Evidence-based answer generation with explicit supporting knowledge paths.
- An evidence-sufficiency mechanism supporting selective answering and abstention.

## Experimental Evaluation

The proposed framework was evaluated on two established multi-hop question-answering benchmarks:

| Dataset | Description |
|---|---|
| ComplexWebQuestions | Complex questions requiring multi-step reasoning and information retrieval |
| WebQuestions | Natural language questions requiring structured knowledge retrieval |

The evaluation investigates the framework's ability to answer complex questions using structured knowledge and supporting evidence.

## Technologies and Tools

- **Programming:** Python
- **LLMs:** Large Language Models for question decomposition, candidate ranking, and answer generation
- **Knowledge Graph:** Wikidata
- **Search Algorithm:** Beam Search
- **Evaluation:** ComplexWebQuestions, WebQuestions
