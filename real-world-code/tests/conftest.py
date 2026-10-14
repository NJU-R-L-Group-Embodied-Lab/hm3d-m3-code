"""Test-only profiles: large/small are geometry fixtures, not hardware types."""

import os
from pathlib import Path

import pytest

os.environ["NAV_CONFIG"] = str(Path(__file__).parent / "fixtures/fleet.json")


@pytest.fixture(autouse=True)
def isolated_runtime(monkeypatch, tmp_path):
    monkeypatch.setenv("M3_HISTORY_DIR", str(tmp_path / "history"))
    monkeypatch.setenv("BOARD_HISTORY_ROOT", str(tmp_path / "board"))

    monkeypatch.setenv("T0_STATIC", "0")
