from __future__ import annotations

import hashlib
import os
import re
from typing import Any, Iterable


def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def stable_doc_id_from_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def normalize_ws(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def batch_items(iterable: list[Any], n: int) -> Iterable[list[Any]]:
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path
