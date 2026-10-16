"""
AegisGraph — Centralized Logging Utility.

Provides a configured logger factory for all backend modules.
"""

import logging
import sys


LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | "
    "%(name)-30s | %(message)s"
)

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

DEFAULT_LEVEL = logging.INFO


def get_logger(
    name: str,
    level: int = DEFAULT_LEVEL,
) -> logging.Logger:
    """
    Return a configured logger for the given module name.

    Parameters
    ----------
    name:
        Typically ``__name__`` of the calling module.

    level:
        Logging level (default: INFO).
    """

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    formatter = logging.Formatter(
        LOG_FORMAT,
        datefmt=DATE_FORMAT,
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.propagate = False

    return logger
