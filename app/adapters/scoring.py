from typing import Optional
import logging
from app.core.config import RunConfig
from app.core.models import TopicEntity, RelationEdge
from app.core.types import CombineRule
import math


logger = logging.getLogger("app.adapters.scoring")

class Scorer:
    """
    Interface for per-hop scoring and cumulative score combination.
    """
    def hop_score(self, subq: str, te: TopicEntity, rel: RelationEdge, ce: TopicEntity) -> float:
        raise NotImplementedError

    def combine(self, cum: float, hop: float, rule: CombineRule) -> float:
        raise NotImplementedError


class DefaultScorer(Scorer):
    """
    Minimal scorer to get the pipeline running.
    - hop_score: returns a constant or a simple heuristic.
      Later you can replace this with your LLM-based scoring.
    - combine: implements add / mul / log accumulation.
    """
    def __init__(self, cfg: RunConfig, *, base_hop: float = 1.0):
        self.cfg = cfg
        self.base_hop = base_hop

    def hop_score(self, subq: str, te: TopicEntity, rel: RelationEdge, ce: TopicEntity) -> float:
        score = self.base_hop
        if rel.label and rel.label.lower() in subq.lower():
            score += 0.1
        if ce.is_value:
            score -= 0.05
        score = max(0.0, score)
        logger.debug(f"hop_score te={te.label} rel={rel.label} ce={ce.label} -> {score}", extra={"session_key": "-"})
        return score

    def combine(self, cum: float, hop: float, rule: CombineRule) -> float:
        if rule == "add":   return cum + hop
        elif rule == "mul": return (cum if cum != 0.0 else 1.0) * max(hop, 1e-9)
        elif rule == "log": return cum + math.log(max(hop, 1e-9))
        else:
            raise ValueError(f"Unknown combine rule: {rule}")
