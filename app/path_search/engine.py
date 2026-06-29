from typing import List
import logging
from app.core.errors import InvariantError
from app.core.models import PathResult, TopicEntity, Branch, SubQuestion, EstimateTime
from app.path_search.branch_expander import BranchExpander
from app.path_search.selectors import BranchSelector


logger = logging.getLogger("app.path_search.engine")


class PathSearchEngine:
    """
    Runs per-path beam search with fixed width K (independent per path).
    """

    def __init__(self, expander: BranchExpander, selector: BranchSelector, k: int):
        if k <= 0:
            raise InvariantError("beam width k must be >= 1")
        self.expander = expander
        self.selector = selector
        self.k = k

    def run_path(self, main_q:str, subqs: List[SubQuestion], initial: TopicEntity, path_id: str | None = None) -> tuple[PathResult, EstimateTime]:
        """
        Initialize beam with one branch at `initial`. For each subquestion:
          - expand all branches
          - if pool empty -> early finish, keep previous beam as final
          - else select top-K -> new beam
        Returns PathResult with final branches.
        """
        logger.info(f"path_start id={path_id} initial={initial.label} subqs={len(subqs)}")
        run_time:EstimateTime = EstimateTime()
        # initial beam
        beam: List[Branch] = [Branch(history=[], frontier=initial, cum_score=0.0, alive=True)]
        prev_nonempty_beam: List[Branch] = beam
        answered_subqs: List[str] = []

        for step_idx, subq in enumerate(subqs):
            # expand every branch in the current beam
            logger.info(f"step_start idx={step_idx} beam_in={len(beam)}")
            pool: List[Branch] = []
            for br in beam:
                branches, expand_est_time = self.expander.expand(
                    branch=br,
                    initial_node=initial,
                    main_q=main_q,
                    previous_sqs=answered_subqs,
                    subq=subq.question,
                    purpose=subq.purpose
                )
                pool.extend(
                    branches
                )
                
                run_time.est_llm += expand_est_time.est_llm
                run_time.est_wiki += expand_est_time.est_wiki
                
            if not pool:
                # early stop for this path: keep last non-empty beam
                logger.warning(f"early_finish at step {step_idx} (no candidates)")
                
                return PathResult(
                    path_id=path_id or f"path_{initial.qid or initial.label}",
                    initial=initial,
                    subqs=subqs,
                    final_branches=list(prev_nonempty_beam),
                ), run_time

            # select top-K branches for next step
            next_beam = self.selector.select_top_k(pool, self.k)
            if len(next_beam) > self.k:
                logger.error("selector returned > K branches")
                raise InvariantError("selector returned more than K branches")
            answered_subqs.append(subq.question)
            prev_nonempty_beam = next_beam
            beam = next_beam

        # normal finish: consumed all subquestions
        logger.info(f"path_finish id={path_id} final_beam={len(beam)}")
        return PathResult(
            path_id=path_id or f"path_{initial.qid or initial.label}",
            initial=initial,
            subqs=subqs,
            final_branches=list(beam),
        ), run_time
