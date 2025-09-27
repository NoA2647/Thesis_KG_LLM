from typing import List, Tuple

from app.core.config import RunConfig
from app.core.models import RelationEdge, TopicEntity
from app.core.errors import KGError
from Data_collection import WikiDataC
from app.wiki_func import abandon_rels


## why we need this?
class KGClient:
    """
    Interface for retrieving (RelationEdge, CandidateEntity) pairs from a frontier TE.
    """
    def relations(self, te: TopicEntity) -> List[Tuple[RelationEdge, TopicEntity]]:
        raise NotImplementedError


class WikidataKG(KGClient):
    """
    Concrete KG adapter wrapping your WikiDataC & helpers.

    - Uses get_all_relations_of_an_entity(qid) to list PIDs & labels
    - Uses get_tail_entities_given_head_and_relation(qid, pid) to fetch candidates
    - Applies your 'abandon_rels' filter when config.remove_unnecessary_rel is True
    - Returns a *flattened* list of (RelationEdge, TopicEntity) pairs ready for scoring
    """

    def __init__(self, wiki: WikiDataC, cfg: RunConfig):
        self.wiki = wiki
        self.cfg = cfg

    def relations(self, te: TopicEntity) -> List[Tuple[RelationEdge, TopicEntity]]:
        if not te.qid:
            # For literal frontiers there are no outgoing Wikidata item relations
            return []

        try:
            rels = self.wiki.get_all_relations_of_an_entity(te.qid)  # [{pid,label}]
        except Exception as e:
            raise KGError(f"Failed to get relations for {te.qid}: {e}") from e

        # optional removal of noisy/unhelpful relations
        if self.cfg.remove_unnecessary_rel:
            rels = [r for r in rels if not abandon_rels(r["label"])]

        out: List[Tuple[RelationEdge, TopicEntity]] = []

        # For each property, fetch its tails (entities or literals)
        for r in rels:
            pid = r.get("pid") or r.get("label")  # fallback to label if pid missing
            label = r.get("label", pid)

            try:
                tails = self.wiki.get_tail_entities_given_head_and_relation(te.qid, pid)
                # tails: [{"label": ..., "qid": ...}] (qid may be None for literals)
            except Exception as e:
                # If a single property fails, skip it but continue
                continue

            edge = RelationEdge(pid=str(pid), label=str(label))
            for t in tails:
                ce_qid = t.get("qid")
                ce_label = t.get("label", "")
                is_value = ce_qid is None
                ce = TopicEntity(qid=ce_qid, label=ce_label, is_value=is_value)
                out.append((edge, ce))

        return out