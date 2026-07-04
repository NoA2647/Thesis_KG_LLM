from typing import Dict, List, Tuple
import logging
import re
import json
from app.core.config import RunConfig
from app.utils import run_llm, extract_answer
from app.prompt_list import REASONING_USER_PROMPT_V2, REASONING_SYS_PROMPT_V2, COT_SYS_PROMPT, COT_USER_PROMPT
from app.core.models import Analyzing, PathResult, Branch
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

    def set_analyzer(self, analyzer: Analyzing):
        self.analyzer = analyzer

    def cot(self, main_q: str) -> Dict:
        self.analyzer = Analyzing()
        start_time = time.time()
        logger.info(f"COT start")
        sys_prompt = COT_SYS_PROMPT
        user_prompt = COT_USER_PROMPT % main_q
        logger.debug(f'COT user_prompt:\n{user_prompt}')
        response, prompt_tokens, completion_tokens = run_llm(
            user_prompt=user_prompt,
            temperature=self.cfg.temperature_reasoning,
            max_tokens=self.cfg.max_length,
            api_keys=self.cfg.api_key,
            engine=self.cfg.llm_type,
            sys_prompt=sys_prompt
        )
        self.analyzer.add_llm_call(prompt_tokens, completion_tokens)
        logger.debug(f'raw result:\n{response}')
        answer = json.loads(response)
        logger.debug(f'extract_answer:\n{answer}')
        logger.info("COT done")
        end_time = time.time()
        self.analyzer.est_llm += (end_time - start_time)
        return {
            "answer": {"ok": True, "answer": answer['final_answer']},
            "paths": [],
            "analyze": self.analyzer
            }
    
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
        response, prompt_tokens, completion_tokens = run_llm(
            user_prompt=user_prompt,
            temperature=self.cfg.temperature_reasoning,
            max_tokens=self.cfg.max_length,
            api_keys=self.cfg.api_key,
            engine=self.cfg.llm_type,
            sys_prompt=sys_prompt
        )
        self.analyzer.add_llm_call(prompt_tokens, completion_tokens)
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
        self.analyzer.est_llm += (end_time - start_time)
        return {"ok": bool(ok), "answer": answer}

# --- helpers --- #


def _format_triplet_dict(t: Dict, indent: str = "  ") -> str:
    """
    Render a single triplet dict in the same literal style used by the
    few-shot examples in REASONING_SYS_PROMPT_V2:

        {
          'head_node':'...',
          'relation_edge':'...',
          'tail_node':'...',
          'tail_node_meta_data': [...]
        }
    """
    return (
        f"{indent}{{\n"
        f"{indent}  'head_node':{t['head_node']!r},\n"
        f"{indent}  'relation_edge':{t['relation_edge']!r},\n"
        f"{indent}  'tail_node':{t['tail_node']!r},\n"
        f"{indent}  'tail_node_meta_data':{t['tail_node_meta_data']!r}\n"
        f"{indent}}}"
    )


def _format_branches_for_llm(branches: List["Branch"], dedupe: bool = True) -> str:
    """
    Flatten all hops across all branches into a single list of triplet dicts
    formatted to match REASONING_SYS_PROMPT_V2's few-shot style:

        [
          {
            'head_node':'Imperial Japanese Army',
            'relation_edge':'allegiance',
            'tail_node':'Emperor of Japan',
            'tail_node_meta_data': []
          },
          ...
        ]

    Notes:
    - No "Path i:" grouping — the examples show one flat triplet list, so we
      match that rather than the old per-branch arrow format.
    - tail_node_meta_data comes from hop.target.meta_data (TopicEntity has no
      separate head-side meta_data field used in the prompt design).
    - dedupe=True drops identical (head, relation, tail) triplets that recur
      across overlapping branches/paths, since the walker often re-derives
      the same hop from multiple starting points — keeps the prompt shorter
      and avoids implying repeated/contradictory evidence.
    """
    triplets: List[Dict] = []
    seen: set[Tuple[str, str, str]] = set()

    for br in branches:
        for hop in br.history:
            head = hop.source.label.replace("\n", " ").strip()
            rel = hop.relation.label.replace("\n", " ").strip()
            tail = hop.target.label.replace("\n", " ").strip()
            meta = hop.target.meta_data or []

            key = (head, rel, tail)
            if dedupe and key in seen:
                continue
            seen.add(key)

            triplets.append({
                "head_node": head,
                "relation_edge": rel,
                "tail_node": tail,
                "tail_node_meta_data": meta,
            })

    if not triplets:
        return "[]"

    blocks = [_format_triplet_dict(t) for t in triplets]
    return "[\n" + ",\n".join(blocks) + "\n]"
