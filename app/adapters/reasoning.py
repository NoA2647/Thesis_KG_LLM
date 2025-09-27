from __future__ import annotations

from typing import Dict, List, Optional

from app.core.config import RunConfig
from app.core.models import PathResult, Branch, Hop
from wiki_func import reasoning as legacy_reasoning  # your existing function


class Reasoner:
    """
    Interface for final reasoning over all path results.
    """
    def reason(self, main_q: str, results: List[PathResult], *, session_key: Optional[str] = None) -> Dict:
        raise NotImplementedError


class LLMReasoner(Reasoner):
    """
    Bridges the new PathResult format to your existing `reasoning(...)` function by
    synthesizing a 'walker-like' object that provides `.format_for_llm()`.
    """

    def __init__(self, cfg: RunConfig, llm_logger=None):
        self.cfg = cfg
        self.llm_logger = llm_logger or _NullLogger()

    def reason(self, main_q: str, results: List[PathResult], *, session_key: Optional[str] = None) -> Dict:
        # 1) Gather all branches across paths
        branches: List[Branch] = []
        for pr in results:
            branches.extend(pr.final_branches)

        # 2) Serialize to the "Knowledge Triplets" text your prompts expect
        triplets_text = _format_branches_for_llm(branches)

        # 3) Build a tiny shim object exposing .format_for_llm() so we can call your legacy function unchanged
        walker_shim = _WalkerShim(triplets_text)

        # 4) Call your existing reasoning(...)
        ok, answer = legacy_reasoning(
            question=main_q,
            walker=walker_shim,
            args={
                "max_length": self.cfg.max_length,
                "temperature_exploration": self.cfg.temperature_exploration,
                "temperature_reasoning": self.cfg.temperature_reasoning,
                "LLM_type": self.cfg.llm_type,
                "opeani_api_keys": self.cfg.openai_api_key,
                # the rest are ignored downstream
                "width": self.cfg.beam_width,
            },
            llm_logger=self.llm_logger,
            session_key=session_key or "",
        )

        return {"ok": bool(ok), "answer": answer}

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


class _WalkerShim:
    """
    Minimal shim with the same .format_for_llm() method your legacy 'reasoning' expects.
    """
    def __init__(self, text: str):
        self._text = text

    def format_for_llm(self) -> str:
        return self._text


class _NullLogger:
    def debug(self, *args, **kwargs): pass
    def info(self, *args, **kwargs): pass
    def warning(self, *args, **kwargs): pass
    def error(self, *args, **kwargs): pass
