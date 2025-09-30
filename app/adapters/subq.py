from typing import List, Optional
import logging
from app.core.config import RunConfig
from app.core.models import TopicEntity
from app.core.errors import SubquestionError
from app.wiki_func import break_question


logger = logging.getLogger("app.adapters.llm")

class SubQGenerator:
    """
    Interface for generating subquestions given main question and initial entity.
    """
    def generate(self, main_q: str, initial: TopicEntity, *, session_key: str) -> List[str]:
        raise NotImplementedError


class LLMSubQGenerator(SubQGenerator):
    """
    Wraps your `break_question(...)` call and returns the cleaned list of subquestions.
    """

    def __init__(self, cfg: RunConfig):
        self.cfg = cfg

    def generate(self, main_q: str, initial: TopicEntity, *, session_key: str) -> List[str]:
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
                llm_logger=logger,
                session_key=session_key,
            )
            logger.info(f"subqs generated: {len(subs)} for {initial.label}", extra={"session_key": session_key})
            return subs or []
        except Exception as e:
            logger.error(f"subq generation failed for {initial.label}: {e}", extra={"session_key": session_key})
            return []
