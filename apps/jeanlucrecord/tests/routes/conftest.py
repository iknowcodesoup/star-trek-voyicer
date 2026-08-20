"""Shared fixtures for the routes/ test suite.

app.py is safe to import directly here: unlike cli.py, it never imports
chatterbox-tts, whisper, or piper_phonemize at module load time, so these
tests need no GPU and no heavy model download.

Every test points filesystem_layout.WORK_DIR at a tmp_path, so nothing here
touches the real work/ directory these stages write to outside of tests.
"""

import pytest
from fastapi.testclient import TestClient

from jeanlucrecord import app as app_module
from jeanlucrecord.infrastructure import filesystem_layout


@pytest.fixture
def work_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(filesystem_layout, "WORK_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def client():
    with TestClient(app_module.app) as test_client:
        yield test_client
