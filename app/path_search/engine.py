from typing import List

from app.core.errors import InvariantError
from app.core.models import PathResult, TopicEntity, Branch
from app.path_search.branch_expander import BranchExpander
from app.path_search.selectors import BranchSelector


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

    def run_path(self, subqs: List[str], initial: TopicEntity, path_id: str | None = None) -> PathResult:
        """
        Initialize beam with one branch at `initial`. For each subquestion:
          - expand all branches
          - if pool empty -> early finish, keep previous beam as final
          - else select top-K -> new beam
        Returns PathResult with final branches.
        """
        # initial beam
        beam: List[Branch] = [Branch(history=[], frontier=initial, cum_score=0.0, alive=True)]
        prev_nonempty_beam: List[Branch] = beam

        for step_idx, subq in enumerate(subqs):
            # expand every branch in the current beam
            pool: List[Branch] = []
            for br in beam:
                pool.extend(self.expander.expand(br, subq))

            if not pool:
                # early stop for this path: keep last non-empty beam
                return PathResult(
                    path_id=path_id or f"path_{initial.qid or initial.label}",
                    initial=initial,
                    subqs=subqs,
                    final_branches=list(prev_nonempty_beam),
                )

            # select top-K branches for next step
            next_beam = self.selector.select_top_k(pool, self.k)
            if len(next_beam) > self.k:
                raise InvariantError("selector returned more than K branches")

            prev_nonempty_beam = next_beam
            beam = next_beam

        # normal finish: consumed all subquestions
        return PathResult(
            path_id=path_id or f"path_{initial.qid or initial.label}",
            initial=initial,
            subqs=subqs,
            final_branches=list(beam),
        )
