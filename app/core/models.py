from dataclasses import dataclass, field
from typing import List, Optional

@dataclass(slots=True)
class TopicEntity:
    """
    A node used as a topic (frontier) or candidate entity.

    qid:    Wikidata item id (e.g., 'Q42') or None for literal values
    label:  Human-readable label or value string
    is_value: True when this represents a literal (date, string, number…)
    """
    
    qid: Optional[str]
    label: str
    is_value: bool = False
    meta_data: List[dict] = field(default_factory=list)
    
    def to_dict(self):
        return {
            "qid": self.qid,
            "label": self.label,
            "is_value": self.is_value,
            "meta_data": self.meta_data
        }

@dataclass(slots=True)
class RelationEdge:
    """
    A relation (property) from a topic entity to a candidate entity.

    pid:   Wikidata property id (e.g., 'P31'); if unknown, set to label as fallback
    label: Human-readable property label
    """
    
    pid: Optional[str]
    label: str
    
    def to_dict(self):
        return {
            "pid": self.pid,
            "label": self.label
        }

@dataclass(slots=True)
class Hop:
    """
    A single traversed edge in a branch.
    """
    
    source: TopicEntity
    relation: RelationEdge
    target: TopicEntity
    hop_score: float = 0.0
    
    def to_dict(self):
        return {
            "source": self.source.to_dict(),
            "relation": self.relation.to_dict(),
            "target": self.target.to_dict(),
            "hop_score": self.hop_score
        }

@dataclass(slots=True)
class Branch:
    """
    A branch (walk) through the KG.

    history:   ordered hops taken so far
    frontier:  the current topic entity at the tip
    cum_score: cumulative score aggregated with the configured rule
    alive:     can be used by higher layers to mark a branch as stalled/invalid
    """
    
    history: List[Hop] = field(default_factory=list)
    frontier: TopicEntity = field(default_factory=lambda: TopicEntity(qid=None, label=""))
    cum_score: float = 0.0
    alive: bool = True
    
    def to_dict(self):
        return {
            "history": [h.to_dict() for h in self.history],
            "frontier": self.frontier.to_dict(),
            "cum_score": self.cum_score,
            "alive": self.alive
        }
    
    def last_hop(self) -> Optional[Hop]:
        return self.history[-1] if self.history[-1] else None
    
    def as_triplets(self) -> List[tuple[str, str, str]]:
        """
        Convenience: (source.label, relation.label, target.label) for LLM prompts.
        """
        
        return [(h.source.label, h.relation.label, h.target.label) for h in self.history]
    
@dataclass(slots=True)
class SubQuestion:
    
    question: str
    purpose: str
    
    def to_dict(self):
        return {
            "question": self.question,
            "purpose": self.purpose,
        }
    
@dataclass(slots=True)
class PathResult:
    """
    Mutable state while searching a single path (for one initial entity).
    """
    
    path_id: str
    initial: TopicEntity
    subqs: List[SubQuestion]
    final_branches: List[Branch] = field(default_factory=list)
    
    def to_dict(self):
        return {
            "path_id": self.path_id,
            "initial": self.initial.to_dict(),
            "subqs": [sq.to_dict() for sq in self.subqs],
            "final_branches": [b.to_dict() for b in self.final_branches],
        }