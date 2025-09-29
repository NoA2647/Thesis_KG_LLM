from dataclasses import dataclass
from typing import Optional

import os
import dotenv
from app.core.errors import ConfigError
from app.core.types import CombineRule

@dataclass(slots=True)
class RunConfig:
    """
    Central configuration for a run.
    Move your old `args` dict fields into here.
    """
    dotenv.load_dotenv()
   
    # Search / beam settings
    beam_width: int = 3
    combine_rule: CombineRule = CombineRule.ADD
    frontier_cap: int = 1

    # KG / pruning
    remove_unnecessary_rel: bool = True
    pruning_enabled: bool = True

    # LLM / API
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    llm_type: str = "gpt-4o-mini"       # LLM Type (e.g. "llama" or "gpt-3.5-turbo" or "gpt-4o" or "gpt-4o-mini")
    temperature_exploration: float = 0.0
    temperature_reasoning: float = 0.0
    max_length: int = 1024

    # Dataset name
    dataset: str = 'cwq'

    # Misc
    rng_seed: Optional[str] = None
    max_paths: Optional[int] = None

    def validate(self) -> None:
        """Raise ConfigError if something is off."""
        
        if self.beam_width <= 0:
            raise ConfigError("beam_width must be >= 1")

        if self.frontier_cap <= 0:
            raise ConfigError("frontier_cap must be >= 1")

        if self.combine_rule not in ("add", "mul", "log"):
            raise ConfigError("combine_rule must be one of: 'add', 'mul', 'log'")

        if not isinstance(self.remove_unnecessary_rel, bool):
            raise ConfigError("remove_unnecessary_rel must be a bool")

        if not isinstance(self.pruning_enabled, bool):
            raise ConfigError("pruning_enabled must be a bool")

        if self.max_length <= 0:
            raise ConfigError("max_length must be a positive integer")

        if self.temperature_exploration < 0 or self.temperature_reasoning < 0:
            raise ConfigError("temperatures must be >= 0")

        if self.max_paths is not None and self.max_paths <= 0:
            raise ConfigError("max_paths, if set, must be >= 1")