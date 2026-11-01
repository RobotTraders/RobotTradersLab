from collections.abc import Iterable
from pathlib import Path


def report_written(paths: Iterable[Path]) -> None:
    for path in paths:
        print(f"Wrote {path}")
