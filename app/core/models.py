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

@dataclass(slots=True)
class RelationEdge:
    """
    A relation (property) from a topic entity to a candidate entity.

    pid:   Wikidata property id (e.g., 'P31'); if unknown, set to label as fallback
    label: Human-readable property label
    """
    
    pid: Optional[str]
    label: str

@dataclass(slots=True)
class Hop:
    """
    A single traversed edge in a branch.
    """
    
    source: TopicEntity
    relation: RelationEdge
    target: TopicEntity
    hop_score: float = 0.0

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
    
    def last_hop(self) -> Optional[Hop]:
        return self.history[-1] if self.history[-1] else None
    
    def as_triplets(self) -> List[tuple[str, str, str]]:
        """
        Convenience: (source.label, relation.label, target.label) for LLM prompts.
        """
        
        return [(h.source.label, h.relation.label, h.target.label) for h in self.history]
    
@dataclass(slots=True)
class PathState:
    """
    Mutable state while searching a single path (for one initial entity).
    """
    
    path_id: str
    initial: TopicEntity
    subqs: List[str]
    step_index: int = 0
    beam: List[Branch] = field(default_factory=list)
    finished: bool = False
    
@dataclass(slots=True)
class PathResult:
    """
    Mutable state while searching a single path (for one initial entity).
    """
    
    path_id: str
    initial: TopicEntity
    subqs: List[str]
    final_branches: List[Branch] = field(default_factory=list)