# cli/app.py  (pseudocode)

from app.core.config import RunConfig
from app.core.models import TopicEntity
from app.adapters.kg_client import WikidataKG         # wraps your WikiDataC
from app.adapters.subq import LLMSubQGenerator       # wraps break_question(...)
from app.adapters.scoring import DefaultScorer       # wraps hop_score/combine
from app.adapters.pruning import NoPruning           # or LLMPruning later
from app.adapters.reasoning import LLMReasoner       # wraps reasoning(...)
from app.path_search.branch_expander import BranchExpander
from app.path_search.selectors import BranchSelector
from app.path_search.engine import PathSearchEngine
from app.path_search.pipeline import PathPipeline

# your existing helpers (unchanged)
from Data_collection import WikiDataC
from utils import prepare_dataset, save_2_json

def main():
    # 0) config (move your old "args" dict into RunConfig fields)
    cfg = RunConfig(
        beam_width=3,
        combine_rule="add",                 # or "mul"/"log" picked before run
        frontier_cap=1,                     # one frontier per branch
        remove_unnecessary_rel=True,
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        llm_type="gpt-4o-mini",
        temperature_exploration=0.0,
        temperature_reasoning=0.0,
        max_length=1024,
        rng_seed=None,
        max_paths=None,                     # optional cap
    )
    cfg.validate()

    # 1) load dataset once
    dataset_name = "cwq"
    data_rows, question_key = prepare_dataset(dataset_name)

    # 2) construct adapters (thin wrappers around your existing code)
    low_level_wiki = WikiDataC()                       # your existing class
    kg = WikidataKG(low_level_wiki, cfg)               # returns (RelationEdge, TopicEntity) pairs
    subq_gen = LLMSubQGenerator(cfg)                   # calls break_question(...)
    scorer = DefaultScorer(cfg)                        # hop_score / combine
    pruner = NoPruning()                               # start simple; swap later
    reasoner = LLMReasoner(cfg)                        # bridges to reasoning(...)

    # 3) construct algorithm pieces
    expander = BranchExpander(kg=kg, pruner=pruner, scorer=scorer, combine_rule=cfg.combine_rule)
    selector = BranchSelector()
    engine = PathSearchEngine(expander=expander, selector=selector, k=cfg.beam_width)
    pipeline = PathPipeline(subq_gen=subq_gen, engine=engine, reasoner=reasoner, cfg=cfg)

    # 4) iterate questions (example: run for one row like your old code)
    #    you can wrap this in argparse to pick indices/limits.
    for row in data_rows[:1]:
        main_q = row[question_key]

        # initial topic entities: dict of {qid: label}; fall back if empty
        initial_map = row.get("qid_topic_entity", {}) or {"Q9696": "John F. Kennedy"}
        initial_entities = [
            TopicEntity(qid=qid, label=label, is_value=False)
            for qid, label in initial_map.items()
        ]

        # 5) run the pipeline (per-path independent beams → reasoning)
        try:
            result = pipeline.run(main_question=main_q, initial_entities=initial_entities)
            # result = {"answer": {...}, "paths": [PathResult, ...]}
        except Exception as e:
            # minimal error handling for the CLI; real code would log & bundle
            print(f"[ERROR] run failed: {e}")
            continue

        # 6) persist output (adapt your save_2_json to new structure)
        #    If you keep the old signature, write a small shim to convert
        #    PathResult branches into the walker/paths format you used before.
        answer = result["answer"]               # whatever your Reasoner returns
        path_results = result["paths"]          # list of PathResult

        # EXAMPLE: save final answer + evidence (adjust to your serializer)
        save_2_json(
            question=main_q,
            results=answer,
            paths=path_results,                 # or a converted form
            file_name=dataset_name
        )

    print("Done.")

if __name__ == "__main__":
    main()
