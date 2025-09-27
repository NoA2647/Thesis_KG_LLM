from typing import List

from app.core.models import Branch, Hop, TopicEntity, RelationEdge
from app.core.types import CombineRule
from app.adapters.kg_client import KGClient
from app.adapters.pruning import Pruner
from app.adapters.scoring import Scorer


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

        if not pairs:
            return []

        # optional pruning
        pairs = self.pruner.prune(subq=subq, te=frontier, pairs=pairs)
        if not pairs:
            return []

        out: List[Branch] = []
        for rel, ce in pairs:
            hop_score = self.scorer.hop_score(subq=subq, te=frontier, rel=rel, ce=ce)
            new_cum = self.scorer.combine(branch.cum_score, hop_score, self.combine_rule)
            new_history = list(branch.history)
            new_history.append(Hop(source=frontier, relation=rel, target=ce, hop_score=hop_score))
            out.append(Branch(history=new_history, frontier=ce, cum_score=new_cum, alive=True))
        return out
