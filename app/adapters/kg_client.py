from typing import List, Tuple
import logging
from app.core.config import RunConfig
from app.core.models import Analyzing, RelationEdge, TopicEntity
from app.core.errors import KGError
from app.adapters.Data_collection import WikiDataC_v2
from app.wiki_func import abandon_rels
import time

logger = logging.getLogger("app.adapters.kg")

class KGClient:
    """
    Interface for retrieving (RelationEdge, CandidateEntity) pairs from a frontier TE.
    """
    def relations(self, te: TopicEntity) -> tuple[List[Tuple[RelationEdge, TopicEntity]],float] :
        raise NotImplementedError


class WikidataKG(KGClient):
    """
    Concrete KG adapter wrapping your WikiDataC_v2 & helpers.

    - Uses get_all_relations_of_an_entity(qid) to list PIDs & labels
    - Uses get_tail_entities_given_head_and_relation(qid, pid) to fetch candidates
    - Applies your 'abandon_rels' filter when config.remove_unnecessary_rel is True
    - Returns a *flattened* list of (RelationEdge, TopicEntity) pairs ready for scoring
    """

    def __init__(self, wiki: WikiDataC_v2, cfg: RunConfig):
        self.wiki = wiki
        self.cfg = cfg
        self.attempt = 3

    def set_analyzer(self, analyzer: Analyzing):
        self.analyzer = analyzer

    def relations(self, te: TopicEntity) -> tuple[List[Tuple[RelationEdge, TopicEntity]],float]:
        start_time = time.time()
        if not te.qid:
            logger.debug("frontier is a literal; no outgoing relations")
            return [], 0.0
        for i in range(self.attempt):
            try:
                rels = self.wiki.get_all_relations_of_an_entity(te.qid)  # [{pid,label}]
                logger.debug(f"relations fetched: {len(rels)} for {te.label}({te.qid})")
                break
            except Exception as e:
                if i != self.attempt - 1:
                    time.sleep(5)
                    continue
                logger.error(f"get_all_relations failed for {te.label}({te.qid}): {e}")
                raise KGError(str(e)) from e

        # optional removal of noisy/unhelpful relations
        if self.cfg.remove_unnecessary_rel:
            before = len(rels)
            rels = [r for r in rels if not abandon_rels(r["label"])]
            logger.debug(f"filtered relations: {before}->{len(rels)}")

        out: List[Tuple[RelationEdge, TopicEntity]] = []

        # For each property, fetch its tails (entities or literals)
        for r in rels:
            pid = r.get("pid") or r.get("label")  # fallback to label if pid missing
            label = r.get("label", pid)

            try:
                tails = self.wiki.get_tail_entities_given_head_and_relation(te.qid, pid)
                # tails: [{"label": ..., "qid": ..., "meta_data": ...}] (qid may be None for literals)
            except Exception as e:
                # If a single property fails, skip it but continue
                logger.warning(f"tails fetch failed for {te.qid} {pid}: {e}")
                continue

            edge = RelationEdge(pid=str(pid), label=str(label))
            for t in tails:
                ce_qid = t.get("qid")
                ce_label = t.get("label", "")
                ce_is_value = ce_qid is None
                ce_meta_data = t.get("meta_data", [])
                ce = TopicEntity(
                    qid=ce_qid,
                    label=ce_label,
                    is_value=ce_is_value,
                    meta_data=ce_meta_data
                )
                out.append((edge, ce))

        logger.debug(f"pairs produced: {len(out)} for {te.qid}")
        end_time = time.time()
        self.analyzer.est_wiki += (end_time - start_time)
        return out