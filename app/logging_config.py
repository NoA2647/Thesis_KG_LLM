import logging
import sys
from typing import Optional

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | SessionKey: %(session_key)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

class _SessionKeyFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # ensure every record has a 'session_key' (formatter-safe)
        if not hasattr(record, "session_key"):
            record.session_key = "-"
        return True

def setup_logging(log_file: Optional[str] = None, level: int = logging.DEBUG) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    root.addFilter(_SessionKeyFilter())

    fmt = logging.Formatter(LOG_FORMAT, DATE_FORMAT)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
