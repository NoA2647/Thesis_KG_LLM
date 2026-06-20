from typing import Dict, List
import logging
import re
import json
from app.core.config import RunConfig
from app.utils import run_llm, extract_answer
from app.prompt_list import REASONING_USER_PROMPT_V2, REASONING_SYS_PROMPT_V2, COT_SYS_PROMPT, COT_USER_PROMPT
from app.core.models import PathResult, Branch
import time

logger = logging.getLogger("app.adapters.llm")

class Reasoner:
    """
    Interface for final reasoning over all path results.
    """
    def reason(self, main_q: str, results: List[PathResult]) -> tuple[Dict, float]:
        raise NotImplementedError


class LLMReasoner(Reasoner):
    """
    Bridges the new PathResult format to your existing `reasoning(...)` function by
    synthesizing a 'walker-like' object that provides `.format_for_llm()`.
    """

    def __init__(self, cfg: RunConfig):
        self.cfg = cfg

    def cot(self, main_q: str) -> tuple[Dict, float]:
        start_time = time.time()
        logger.info(f"COT start")
        sys_prompt = COT_SYS_PROMPT
        user_prompt = COT_USER_PROMPT % main_q
        logger.debug(f'COT user_prompt:\n{user_prompt}')
        response = run_llm(
            user_prompt=user_prompt,
            temperature=self.cfg.temperature_reasoning,
            max_tokens=self.cfg.max_length,
            api_keys=self.cfg.api_key,
            engine=self.cfg.llm_type,
            sys_prompt=sys_prompt
        )
        
        logger.debug(f'raw result:\n{response}')
        answer = json.loads(response)
        logger.debug(f'extract_answer:\n{answer}')
        logger.info("COT done")
        end_time = time.time()
        cot_est_time:float = end_time - start_time
        return {
            "answer": {"ok": True, "answer": answer['final_answer']},
            "paths": []
            }, cot_est_time
    
    def reason(self, main_q: str, results: List[PathResult]) -> tuple[Dict, float]:
        # 1) Gather all branches across paths
        start_time = time.time()
        branches: List[Branch] = []
        for pr in results:
            branches.extend(pr.final_branches)

        # 2) Serialize to the "Knowledge Triplets" text your prompts expect
        triplets_text = _format_branches_for_llm(branches)

        # 3) Call your existing reasoning(...)
        logger.info(f"reasoning start: branches={len(branches)}")
        
        sys_prompt = REASONING_SYS_PROMPT_V2
        user_prompt = REASONING_USER_PROMPT_V2 % (main_q, triplets_text)
        
        logger.debug(f'reasoning user_prompt:\n{user_prompt}')
        response = run_llm(
            user_prompt=user_prompt,
            temperature=self.cfg.temperature_reasoning,
            max_tokens=self.cfg.max_length,
            api_keys=self.cfg.api_key,
            engine=self.cfg.llm_type,
            sys_prompt=sys_prompt
        )
        logger.debug(f'raw result:\n{response}')
        result = json.loads(response)
        logger.debug(f'extract_answer:\n{result}')
        if result is not None:
            ok = True
            answer = result
        else:
            ok = False
            answer = None
        
        logger.info("reasoning done")
        end_time = time.time()
        reasoning_est_time:float = end_time - start_time
        return {"ok": bool(ok), "answer": answer}, reasoning_est_time

# --- helpers --- #

def _format_branches_for_llm(branches: List[Branch]) -> str:
    """
    Produce the same style of "Knowledge Triplets" text your walker.format_for_llm() returns.
    We group by branch and print each hop 'TE --REL--> CE'.
    """
    lines: List[str] = []
    for i, br in enumerate(branches, start=1):
        lines.append(f"Path {i}:")
        for hop in br.history:
            # ensure labels are strings without newlines
            s = hop.source.label.replace("\n", " ")
            r = hop.relation.label.replace("\n", " ")
            t = hop.target.label.replace("\n", " ")
            lines.append(f"{s} --{r}--> {t}")
        lines.append("")  # blank line
    return "\n".join(lines)
