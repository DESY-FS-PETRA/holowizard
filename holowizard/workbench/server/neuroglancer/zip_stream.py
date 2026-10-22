"""Sink for writing a zip file piece by piece into a streamed response."""

from typing import List


class ZipStream:
    """File-like sink for zipfile that hands the written bytes to a generator."""

    def __init__(self):
        self.parts: List[bytes] = []

    def write(self, b) -> int:
        self.parts.append(bytes(b))
        return len(b)

    def flush(self):
        pass

    def take(self) -> bytes:
        data, self.parts = b"".join(self.parts), []
        return data
