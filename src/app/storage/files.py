from __future__ import annotations

import logging
import os
import tempfile


logger = logging.getLogger(__name__)


def write_temp_file(data: bytes, suffix: str = ".pdf") -> str:
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        return tmp.name


def remove_temp_file(path: str) -> None:
    if not path:
        return
    try:
        os.unlink(path)
    except OSError:
        logger.warning("Failed to remove temp upload file: %s", path)
