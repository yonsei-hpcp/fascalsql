from __future__ import annotations

from pathlib import Path
from typing import Iterator, List


def _clean_tbl_field_bytes(field: bytes) -> str:
    try:
        text = field.decode("utf-8")
    except UnicodeDecodeError:
        text = field.decode("utf-8", errors="ignore")
        if not text:
            text = field.decode("latin-1", errors="ignore")
    return "".join(ch for ch in text if ch >= " " or ch == "\t")


def parse_tbl_line(line: str | bytes) -> List[str]:
    if isinstance(line, bytes):
        parts = line.rstrip(b"\r\n").split(b"|")
        if parts and parts[-1] == b"":
            parts = parts[:-1]
        return [_clean_tbl_field_bytes(part) for part in parts]

    parts = line.rstrip("\r\n").split("|")
    if parts and parts[-1] == "":
        parts = parts[:-1]
    return parts


def iter_tbl_rows(tbl_path: Path) -> Iterator[List[str]]:
    with tbl_path.open("rb") as fin:
        for line in fin:
            yield parse_tbl_line(line)


def iter_tbl_row_chunks(tbl_path: Path, chunk_size: int = 100_000) -> Iterator[List[List[str]]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    chunk: List[List[str]] = []
    for row in iter_tbl_rows(tbl_path):
        chunk.append(row)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []

    if chunk:
        yield chunk
