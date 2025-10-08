from typing import Dict, List
import logging
from app.core.config import RunConfig
from app.core.models import TopicEntity, PathResult, SubQuestion
from app.adapters.subq import SubQGenerator
from app.adapters.reasoning import Reasoner
from app.path_search.engine import PathSearchEngine

logger = logging.getLogger("app.pipeline")

class PathPipeline:
    """
    Orchestrates:
      - subquestion generation per initial entity,
      - per-path beam search,
      - final reasoning over all path results.
    """

    def __init__(self, subq_gen: SubQGenerator, engine: PathSearchEngine, reasoner: Reasoner, cfg: RunConfig):
        self.subq_gen = subq_gen
        self.engine = engine
        self.reasoner = reasoner
        self.cfg = cfg

    def run(self, main_question: str, initial_entities: List[TopicEntity], *, session_key: str) -> Dict:
        # Optional cap on number of paths to process
        initials = initial_entities[: self.cfg.max_paths] if self.cfg.max_paths else list(initial_entities)
        logger.info(f"pipeline_start initials={len(initials)} qlen={len(main_question)}", extra={"session_key": session_key})

        path_results: List[PathResult] = []
        for idx, initial in enumerate(initials):
            # 1) generate subquestions for this path
            subqs:List[SubQuestion] = self.subq_gen.generate(main_q=main_question, initial=initial, session_key=session_key)
            if not subqs:
                # No subquestions -> degenerate path result with empty final branches
                logger.warning(f"No subqs for {initial.label}", extra={"session_key": session_key})
                path_results.append(
                    PathResult(
                        path_id=f"path_{idx}_{initial.qid or initial.label}",
                        initial=initial,
                        subqs=[],
                        final_branches=[],
                    )
                )
                continue

            # 2) run per-path beam search
            res = self.engine.run_path(main_q=main_question,
                                       subqs=subqs,
                                       initial=initial,
                                       path_id=f"path_{idx}_{initial.qid or initial.label}",
                                       session_key=session_key)
            path_results.append(res)

        # 3) run reasoning over all paths
        answer = self.reasoner.reason(main_q=main_question, results=path_results, session_key=session_key)
        logger.info("pipeline_done", extra={"session_key": session_key})
        return {"answer": answer, "paths": path_results}
