import logging
from typing import List
from app.core.models import Branch

logger = logging.getLogger("app.path_search.selector")

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
        out = sorted_cands[:k]
        logger.info(f"select_top_k candidates={len(candidates)} -> kept={len(out)}", extra={"session_key": "-"})
        return out
