from typing import List, Optional

from app.core.config import RunConfig
from app.core.models import TopicEntity
from app.core.errors import SubquestionError
from wiki_func import break_question  # your existing function

class SubQGenerator:
    """
    Interface for generating subquestions given main question and initial entity.
    """
    def generate(self, main_q: str, initial: TopicEntity, *, session_key: Optional[str] = None) -> List[str]:
        raise NotImplementedError


class LLMSubQGenerator(SubQGenerator):
    """
    Wraps your `break_question(...)` call and returns the cleaned list of subquestions.
    """

    def __init__(self, cfg: RunConfig, llm_logger=None):
        self.cfg = cfg
        self.llm_logger = llm_logger  # optional; you can inject your logger later

    def generate(self, main_q: str, initial: TopicEntity, *, session_key: Optional[str] = None) -> List[str]:
        try:
            subs = break_question(
                question=main_q,
                entity_name=initial.label,
                args={
                    "max_length": self.cfg.max_length,
                    "temperature_exploration": self.cfg.temperature_exploration,
                    "temperature_reasoning": self.cfg.temperature_reasoning,
                    "LLM_type": self.cfg.llm_type,
                    "opeani_api_keys": self.cfg.openai_api_key,  # keep your current key name for compatibility
                    # fields below are ignored by break_question but harmless
                    "width": self.cfg.beam_width,
                    "remove_unnecessary_rel": self.cfg.remove_unnecessary_rel,
                },
                llm_logger=self.llm_logger or _NullLogger(),
                session_key=session_key or "",
            )
            return subs or []
        except Exception as e:
            # Surface as empty list; engine will end path gracefully.
            return []


class _NullLogger:
    def debug(self, *args, **kwargs):
        pass
    def info(self, *args, **kwargs):
        pass
    def warning(self, *args, **kwargs):
        pass
    def error(self, *args, **kwargs):
        pass
