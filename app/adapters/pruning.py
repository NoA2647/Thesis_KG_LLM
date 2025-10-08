from typing import List, Tuple
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
        main_q: str,
        current_sq: str,
        previous_sqs: List[str],
        purpose: str,
        initial_node: TopicEntity,
        current_node: TopicEntity,
        previous_hops: List[Hop],
        pairs: List[Tuple[RelationEdge, TopicEntity]],
        session_key: str
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
        main_q: str,
        current_sq: str,
        previous_sqs: List[str],
        purpose: str,
        initial_node: TopicEntity,
        current_node: TopicEntity,
        previous_hops: List[Hop],
        pairs: List[Tuple[RelationEdge, TopicEntity]],
        session_key: str
    ) -> List[Hop]:

        if not current_node.qid or not pairs:
            return [Hop(current_node, rel, ce, 0.0) for rel, ce in pairs]
        
        hops = relation_entity_search_prune(main_q=main_q,
                                 sub_question=current_sq,
                                 previous_sub_questions=previous_sqs,
                                 purpose=purpose,
                                 initial_node=initial_node,
                                 current_node=current_node,
                                 previous_hops=previous_hops,
                                 relation_entity_pairs=pairs,
                                 configs=self.cfg,
                                 llm_logger=logger,
                                 session_key=session_key)
        logger.debug(f"LLM pruning applied; hops={len(hops)}", extra={"session_key": session_key})

        return hops
