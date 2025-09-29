from typing import List
import logging
from app.core.models import Branch, Hop, TopicEntity, RelationEdge
from app.core.types import CombineRule
from app.adapters.kg_client import KGClient
from app.adapters.pruning import Pruner
from app.adapters.scoring import Scorer

logger = logging.getLogger("app.path_search.expander")

class BranchExpander:
    """
    Expands a single Branch by one step using:
      - KGClient to fetch (RelationEdge, CandidateEntity) pairs,
      - Pruner to optionally filter pairs,
      - Scorer to compute hop_score and update cum_score.
    """

    def __init__(self, kg: KGClient, pruner: Pruner, scorer: Scorer, combine_rule: CombineRule):
        self.kg = kg
        self.pruner = pruner
        self.scorer = scorer
        self.combine_rule = combine_rule

    def expand(self, branch: Branch, subq: str) -> List[Branch]:
        """
        Returns a list of new branches produced by taking one hop from `branch.frontier`.
        May return [] if no candidates or on adapter failures (treated as no expansion).
        """
        frontier: TopicEntity = branch.frontier
        pairs: List[tuple[RelationEdge, TopicEntity]] = self.kg.relations(frontier)
        logger.debug(f"expand frontier={frontier.label} pairs={len(pairs)}", extra={"session_key": "-"})

        if not pairs:
            return []

        hops: List[Hop] = self.pruner.prune(subq=subq, te=frontier, pairs=pairs)
        if not hops:
            return []

        out: List[Branch] = []
        for hop in hops:
            new_cum = self.scorer.combine(branch.cum_score, hop.hop_score, self.combine_rule)
            new_history = list(branch.history)
            new_history.append(hop)
            out.append(Branch(history=new_history, frontier=hop.target, cum_score=new_cum, alive=True))
        logger.debug(f"expanded branches: {len(out)}", extra={"session_key": "-"})
        return out
