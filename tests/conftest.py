import os

import pytest

os.environ.setdefault("DRIVEMEMORY_DISABLE_LLM", "1")


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DRIVEMEMORY_DB", str(tmp_path / "test.db"))
    monkeypatch.chdir(tmp_path)
