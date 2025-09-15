class ConfigError(Exception):
    """Invalid or inconsistent configuration."""


class InvariantError(Exception):
    """A program invariant was violated (e.g., beam > K)."""


# --- Adapters / IO layer errors --- #
class KGError(Exception):
    """Base for knowledge-graph related errors."""


class KGTimeout(KGError):
    """KG request timed out."""


class KGUnavailable(KGError):
    """KG endpoint is unavailable."""


class KGMalformed(KGError):
    """KG returned malformed data."""


class SubquestionError(Exception):
    """Failed to generate subquestions."""


class ScoringError(Exception):
    """Failed to score a hop."""


class ReasoningError(Exception):
    """Failed during final reasoning."""
