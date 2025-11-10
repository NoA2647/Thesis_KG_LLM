# app/logging_config.py
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional
from contextvars import ContextVar
from pathlib import Path

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | SessionKey: %(session_key)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

CURRENT_SESSION_KEY: ContextVar[str] = ContextVar("session_key", default="-")

def set_session_key(value: str) -> None:
    CURRENT_SESSION_KEY.set(value)

def _install_record_factory():
    old_factory = logging.getLogRecordFactory()
    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        if not hasattr(record, "session_key"):
            record.session_key = CURRENT_SESSION_KEY.get()
        return record
    logging.setLogRecordFactory(record_factory)

def setup_logging(
    log_file: Optional[str] = "app/run.log",
    level: int = logging.DEBUG,
    *,
    console: bool = False,            # default: no console logs
    rotate: bool = True,              # rotate to keep files tidy
) -> None:
    _install_record_factory()

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    fmt = logging.Formatter(LOG_FORMAT, DATE_FORMAT)

    # ensure folder exists
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)

        if rotate:
            fh = RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10 MB
                backupCount=5,
                encoding="utf-8",
                delay=True,                  # open file lazily
            )
        else:
            fh = logging.FileHandler(log_file, encoding="utf-8", delay=True)

        fh.setFormatter(fmt)
        root.addHandler(fh)

    # add console handler only if explicitly requested
    if console:
        ch = logging.StreamHandler()  # defaults to sys.stderr
        ch.setFormatter(fmt)
        root.addHandler(ch)

    # quiet noisy libs if desired (still go to file only)
    logging.getLogger("httpx").setLevel(logging.INFO)
    logging.getLogger("httpcore").setLevel(logging.INFO)
    logging.getLogger("openai").setLevel(logging.INFO)
