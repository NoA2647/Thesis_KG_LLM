import uuid, logging
from app.logging_config import setup_logging, set_session_key
from app.core.config import RunConfig
from app.core.models import TopicEntity
from app.adapters.kg_client import WikidataKG
from app.adapters.subq import LLMSubQGenerator
from app.adapters.scoring import DefaultScorer
from app.adapters.pruning import LLMPruning
from app.adapters.reasoning import LLMReasoner
from app.path_search.branch_expander import BranchExpander
from app.path_search.selectors import BranchSelector
from app.path_search.engine import PathSearchEngine
from app.path_search.pipeline import PathPipeline
from tqdm import tqdm
from app.adapters.Data_collection import WikiDataC_v2
from app.utils import prepare_dataset, save_2_json
from app.adapters.graph_drawer import build_and_draw_graph

def main():
    setup_logging(log_file="app/run.log", level=logging.DEBUG)
    logger = logging.getLogger("app.main")

    session_key = str(uuid.uuid4())
    set_session_key(session_key)  
    
    
    logger.info("run_start")
    
    cfg = RunConfig()
    cfg.validate()

    data_rows, question_key, id_key = prepare_dataset(cfg.dataset)
    logger.info(f"\'{cfg.dataset}\' dataset loaded rows={len(data_rows)}")

    # 2) construct adapters (thin wrappers around your existing code)
    low_level_wiki = WikiDataC_v2()
    kg = WikidataKG(low_level_wiki, cfg)
    subq_gen = LLMSubQGenerator(cfg)
    scorer = DefaultScorer(cfg)
    pruner = LLMPruning(cfg)
    reasoner = LLMReasoner(cfg)

    # 3) construct algorithm pieces
    expander = BranchExpander(kg=kg, pruner=pruner, scorer=scorer, combine_rule=cfg.combine_rule)
    selector = BranchSelector()
    engine = PathSearchEngine(expander=expander, selector=selector, k=cfg.beam_width)
    pipeline = PathPipeline(subq_gen=subq_gen, engine=engine, reasoner=reasoner, cfg=cfg)

    # 4) iterate questions
    #    you can wrap this in argparse to pick indices/limits.
    for idx, row in enumerate(tqdm(data_rows, total=len(data_rows))):
        main_q = row[question_key]
        data_id = row.get(id_key, f'{main_q}_{idx}')
        # initial topic entities: dict of {qid: label}
        initial_map = row.get("qid_topic_entity", {}) # {"Q9696": "John F. Kennedy"}
        if not initial_map:
            logger.warning(f"no_initial_entities row={idx}; COT generating")
            result = reasoner.cot(main_q=main_q)
        
        else:
            initial_entities = [
                TopicEntity(qid=qid, label=label, is_value=False)
                for qid, label in initial_map.items()
            ]
            
            logger.info(f"row_start idx={idx} initials={len(initial_entities)} qlen={len(main_q)}")

            # 5) run the pipeline (per-path independent beams → reasoning)
            try:
                result = pipeline.run(main_question=main_q, initial_entities=initial_entities)
                # result = {"answer": {...}, "paths": [PathResult, ...]}
            except Exception as e:
                logger.exception(f"pipeline_failed row={idx}, data_id={data_id}")
                continue

        answer = result["answer"]               # return answer={ok:bool, answer:{"sufficient": answered, "explanation": explanation, "final_answer": answer}}
        path_results = result["paths"]          # list of PathResult
        analyze = result['analyze']
        try:
            payload = save_2_json(data_id, question=main_q, results=answer, paths=path_results, analyze=analyze, file_name=cfg.dataset)
            build_and_draw_graph(data=payload, save_path=payload['id'])
            logger.info(f"row_saved idx={idx}, data_id={data_id}")
        except Exception:
            logger.exception(f"save_failed row={idx}, data_id={data_id}")

    logger.info("run_done")

    print("Done.")

if __name__ == "__main__":
    main()
