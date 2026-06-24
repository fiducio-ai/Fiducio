"""Lightweight logging helpers for Fiducio.

Library code uses the standard :mod:`logging` module and never prints to stdout.
Applications remain in full control of handlers and levels.
"""

from __future__ import annotations

import logging

_LOGGER_NAME = "fiducio"


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger under the ``fiducio`` namespace.

    Parameters
    ----------
    name:
        Optional dotted suffix (typically ``__name__``). When ``None`` the root
        ``fiducio`` logger is returned.
    """
    if not name or name == _LOGGER_NAME:
        return logging.getLogger(_LOGGER_NAME)
    if name.startswith(_LOGGER_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_LOGGER_NAME}.{name}")
