"""Shared fixtures for the routes/ test suite.

app.py is safe to import directly here: unlike cli.py, it never imports
chatterbox-tts, whisper, or piper_phonemize at module load time, so these
tests need no GPU and no heavy model download.

Every test points filesystem_layout.WORK_DIR at a tmp_path, so nothing here
touches the real work/ directory these stages write to outside of tests.
"""

import pytest
from fastapi.testclient import TestClient

from voice_factory import app as app_module
from voice_factory.infrastructure import filesystem_layout


@pytest.fixture
def work_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(filesystem_layout, "WORK_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def missing_work_dir(tmp_path, monkeypatch):
    """A WORK_DIR that is not there, which is a broken deployment.

    tmp_path always exists, so the fault this fixture reproduces -- the one
    that made GET /videos answer 200 with an empty list on 2026-08-20 -- needs
    a path one level below it that nothing creates.
    """
    absent = tmp_path / "nope"
    monkeypatch.setattr(filesystem_layout, "WORK_DIR", absent)
    return absent


@pytest.fixture
def client():
    with TestClient(app_module.app) as test_client:
        yield test_client
