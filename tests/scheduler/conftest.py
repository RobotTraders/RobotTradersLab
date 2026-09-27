from collections.abc import Callable
from pathlib import Path

import pytest


@pytest.fixture
def create_registry(tmp_path: Path) -> Callable[[str], Path]:
    def _create_registry(content: str) -> Path:
        registry_file = tmp_path / "scheduler.toml"
        registry_file.write_text(content)
        return registry_file

    return _create_registry
