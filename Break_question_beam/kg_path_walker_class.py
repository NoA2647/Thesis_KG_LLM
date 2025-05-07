from typing import List, Tuple, Literal
import math

Triple = Tuple[str, str, str]
ScoredTriple = Tuple[str, str, str, float]
Path = List[Triple]
ScoredPath = Tuple[float, Path]

class KGPathWalker:
    def __init__(self, root_entities: List[str], scoring_mode: Literal["add", "mul", "log"] = "mul"):
        """
        scoring_mode: 
            - "add": adds scores
            - "mul": multiplies scores (probabilities, sum-to-1)
            - "log": log-multiplies scores to avoid underflow
        """
        self.paths: List[ScoredPath] = [(1.0 if scoring_mode != "log" else 0.0, [(e, "_ROOT_", e)]) for e in root_entities]
        self.scoring_mode = scoring_mode

    def extend_paths(self, new_scored_triples: List[ScoredTriple]):
        """Extend each path by matching last tail to subject, and accumulate score accordingly."""
        extended = []
        for path_score, path in self.paths:
            tail = path[-1][2]
            matched = [t for t in new_scored_triples if t[0] == tail]
            if matched:
                for subj, rel, obj, triple_score in matched:
                    new_path = path + [(subj, rel, obj)]
                    new_score = self._update_score(path_score, triple_score)
                    extended.append((new_score, new_path))
            else:
                extended.append((path_score, path))  # no change
        self.paths = extended

    def _update_score(self, path_score: float, triple_score: float) -> float:
        if self.scoring_mode == "add":
            return path_score + triple_score
        elif self.scoring_mode == "mul":
            return path_score * triple_score
        elif self.scoring_mode == "log":
            return path_score + math.log(triple_score + 1e-10)  # prevent log(0)
        else:
            raise ValueError("Invalid scoring mode")

    def beam_search(self, k: int):
        """Keep top-k paths by score (or log-score)."""
        self.paths.sort(reverse=True, key=lambda x: x[0])
        self.paths = self.paths[:k]

    def get_all_paths(self) -> List[ScoredPath]:
        return self.paths

    def format_for_llm(self) -> str:
        """Return all paths as string for LLM input."""
        lines = []
        for i, (score, path) in enumerate(self.paths):
            display_score = math.exp(score) if self.scoring_mode == "log" else score
            lines.append(f"Path {i+1}:")
            for subj, rel, obj in path:
                if rel != "_ROOT_":
                    lines.append(f"{subj} --{rel}--> {obj}")
            lines.append("")  # blank line
        return "\n".join(lines)

    def print_paths(self):
        """Return all paths as string with score for Debugging."""
        lines = []
        for i, (score, path) in enumerate(self.paths):
            display_score = math.exp(score) if self.scoring_mode == "log" else score
            lines.append(f"Path {i+1} (score: {display_score:.4f}):")
            for subj, rel, obj in path:
                if rel != "_ROOT_":
                    lines.append(f"{subj} --{rel}--> {obj}")
            lines.append("")  # blank line
        return "\n".join(lines)
