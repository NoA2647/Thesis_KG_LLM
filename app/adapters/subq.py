from typing import List
import json 
import logging
from app.core.config import RunConfig
from app.core.models import TopicEntity, SubQuestion
from app.prompt_list import BREAK_QUESTION_USER_PROMPT, BREAK_QUESTION_SYS_PROMPT
from app.utils import run_llm
from app.wiki_func import break_question


logger = logging.getLogger("app.adapters.llm")

class SubQGenerator:
    """
    Interface for generating subquestions given main question and initial entity.
    """
    def generate(self, main_q: str, initial: TopicEntity, *, session_key: str) -> List[SubQuestion]:
        raise NotImplementedError


class LLMSubQGenerator(SubQGenerator):
    """
    Wraps your `break_question(...)` call and returns the cleaned list of subquestions.
    """

    def __init__(self, cfg: RunConfig):
        self.cfg = cfg

    def generate(self, main_q: str, initial: TopicEntity, *, session_key: str) -> List[SubQuestion]:
        subqs:List[SubQuestion] = []
        try:
            user_prompt = BREAK_QUESTION_USER_PROMPT % (main_q, initial.label)
            logger.debug(f'break_question prompt:\n{user_prompt}', extra={'session_key': session_key})
            result = run_llm(user_prompt=user_prompt,
                            temperature=self.cfg.temperature_exploration,
                            max_tokens=self.cfg.max_length,
                            openai_api_keys=self.cfg.openai_api_key,
                            sys_prompt=BREAK_QUESTION_SYS_PROMPT,
                            engine=self.cfg.llm_type)
            logger.debug(f'Raw results:\n{result}', extra={'session_key': session_key})
            subs = json.loads(result)['steps']
            logger.debug(f'All sub_questions:\n{subs}', extra={'session_key': session_key})
            subqs = [SubQuestion(s['sub_question'], s['purpose']) for s in subs]
            logger.info(f"subqs generated: {len(subs)} for {initial.label}", extra={"session_key": session_key})
            return subqs
        except Exception as e:
            logger.error(f"subq generation failed for {initial.label}: {e}", extra={"session_key": session_key})
            return subqs
