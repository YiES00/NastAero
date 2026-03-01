"""Global solver configuration and logging setup."""
import logging
import sys

LOG_FORMAT = "%(levelname)-8s %(name)-20s %(message)s"

def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=LOG_FORMAT,
        stream=sys.stdout,
    )

logger = logging.getLogger("nastaero")
