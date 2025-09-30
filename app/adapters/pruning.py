from typing import List, Tuple, Optional
import logging
from app.core.models import TopicEntity, RelationEdge, Hop
from app.wiki_func import relation_entity_search_prune
from app.core.config import RunConfig

logger = logging.getLogger("app.adapters.pruning")

class Pruner:
    """
    Interface for optional relation/entity pruning before scoring.
    """
    def prune(
        self,
        subq: str,
        te: TopicEntity,
        pairs: List[Tuple[RelationEdge, TopicEntity]],
        session_key: str,
    ) -> List[Hop]:
        raise NotImplementedError


class LLMPruning(Pruner):
    """
    Default: LLM prunning realtion-nodes.
    """
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
    
    def prune(
        self,
        subq: str,
        te: TopicEntity,
        pairs: List[Tuple[RelationEdge, TopicEntity]],
        session_key: str
    ) -> List[Hop]:

        if not te.qid or not pairs:
            return [Hop(te, rel, ce, 0.0) for rel, ce in pairs]
        
        hops = relation_entity_search_prune(te, subq, pairs, {
                "max_length": self.cfg.max_length,
                "temperature_exploration": self.cfg.temperature_exploration,
                "temperature_reasoning": self.cfg.temperature_reasoning,
                "LLM_type": self.cfg.llm_type,
                "opeani_api_keys": self.cfg.openai_api_key,  # keep your current key name for compatibility
                # fields below are ignored by break_question but harmless
                "width": self.cfg.beam_width,
                "remove_unnecessary_rel": self.cfg.remove_unnecessary_rel,
            }, logger, session_key)
        logger.debug(f"LLM pruning applied; hops={len(hops)}", extra={"session_key": session_key})

        return hops
