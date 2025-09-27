from typing import List

from app.core.models import Branch


class BranchSelector:
    """
    Selects top-K branches *within a path* deterministically.
    Sort key:
      - cum_score DESC
      - last_hop.target.qid ASC (None -> "")
      - last_hop.relation.pid ASC
    """

    def select_top_k(self, candidates: List[Branch], k: int) -> List[Branch]:
        if not candidates:
            return []

        def _key(b: Branch):
            last = b.last_hop()
            ce_qid = (last.target.qid or "") if last else ""
            pid = (last.relation.pid or "") if last else ""
            return (b.cum_score, ce_qid, pid)

        # Sort by primary desc, tie-breakers asc
        sorted_cands = sorted(candidates, key=_key, reverse=True)
        return sorted_cands[:k]
