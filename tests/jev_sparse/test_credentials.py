"""Hermetic credential tests use a temporary file and never read the user's key."""

import os
import sys

import gradio as gr
import pytest

from modules_forge.jev_sparse import common, credentials, ui


def test_key_roundtrip_is_not_plaintext_on_windows(tmp_path):
    path = tmp_path / "key"
    key = "apikey_" + "test-credential-" * 3
    credentials.save_key(key, path)
    assert credentials.read_saved_key(path) == key
    if os.name == "nt":
        assert key not in path.read_text()
    else:
        assert path.stat().st_mode & 0o777 == 0o600
    credentials.save_key(key + "new", path)
    assert credentials.read_saved_key(path) == key + "new"


def test_invalid_key_does_not_replace_saved_value(tmp_path):
    path = tmp_path / "key"
    path.write_text("unchanged")
    with pytest.raises(ValueError):
        credentials.save_key("not an API key", path)
    assert path.read_text() == "unchanged"


def test_key_cannot_be_saved_under_repository():
    from pathlib import Path

    with pytest.raises(ValueError, match="リポジトリの外側"):
        credentials.save_key("apikey_" + "a" * 32, Path(credentials.__file__).resolve().parents[2] / "tmp/jev-api-key")


def test_ui_never_echoes_key(monkeypatch, tmp_path):
    monkeypatch.setattr(ui, "save_key", lambda key: credentials.save_key(key, tmp_path / "key"))
    secret = "apikey_" + "a" * 32
    cleared, message = ui.save_from_ui(secret)
    assert cleared == "" and secret not in message
    monkeypatch.setattr(ui, "credential_path", lambda: tmp_path / "key")
    with gr.Blocks() as app:
        field, _, _ = ui.credential_controls("test")
    assert field.value == "" and field.type == "password"
    assert secret not in str(app.get_config_file())


def test_valid_low_confidence_choice_is_not_replaced_with_dense(monkeypatch, mock_sdk):
    mock_sdk(choice=25, confidence=0.12)
    client = common.JevClient(
        sys.executable, environment={"AIKIMI_JEV_ALLOW_CLOUD": "1", "TYPESAFE_API_KEY": "test-key"}
    )
    assert client.decide({}, {"0": (25.0, 50.0, 75.0, 100.0)}, 100.0) == {"0": 25.0}
    client.close()
    assert client.last_diagnostics["0"] == {
        "proposed_keep_percent": 25.0,
        "confidence": 0.12,
        "applied_keep_percent": 25.0,
        "confidence_policy": "report_only",
    }
