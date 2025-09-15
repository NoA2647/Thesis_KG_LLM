from typing import List, Tuple

from app.core.config import RunConfig
from app.core.models import RelationEdge, TopicEntity
from app.core.errors import KGError
from Data_collection import WikiDataC
from app.wiki_func import abandon_rels  # reuse your filter
