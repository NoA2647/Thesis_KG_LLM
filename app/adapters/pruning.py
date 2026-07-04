from typing import List, Tuple
import logging
from app.core.models import Analyzing, TopicEntity, RelationEdge, Hop
from app.wiki_func import relation_entity_search_prune
from app.core.config import RunConfig
import time

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
        pairs: List[Tuple[RelationEdge, TopicEntity]]
    ) -> tuple[List[Hop], float]:
        raise NotImplementedError


class LLMPruning(Pruner):
    """
    Default: LLM prunning realtion-nodes.
    """
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
    
    def set_analyzer(self, analyzer: Analyzing):
        self.analyzer = analyzer
    
    def prune(
        self,
        main_q: str,
        current_sq: str,
        previous_sqs: List[str],
        purpose: str,
        initial_node: TopicEntity,
        current_node: TopicEntity,
        previous_hops: List[Hop],
        pairs: List[Tuple[RelationEdge, TopicEntity]]
    ) -> tuple[List[Hop], float]:

        if not current_node.qid or not pairs:
            return [Hop(current_node, rel, ce, 0.0) for rel, ce in pairs], 0.0
        start_time = time.time()
        hops = relation_entity_search_prune(
            main_q=main_q,
            sub_question=current_sq,
            previous_sub_questions=previous_sqs,
            purpose=purpose,
            initial_node=initial_node,
            current_node=current_node,
            previous_hops=previous_hops,
            relation_entity_pairs=pairs,
            configs=self.cfg,
            analyzer=self.analyzer,
            llm_logger=logger
        )
        logger.debug(f"LLM pruning applied; hops={len(hops)}")
        end_time = time.time()
        self.analyzer.est_llm += (end_time - start_time)
        return hops
