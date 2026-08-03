"""Tests for the config settings (env-driven, no secrets needed)."""

from deepsight.config import Settings


def test_defaults():
    s = Settings(_env_file=None)
    assert s.host == "127.0.0.1"
    assert s.port == 8080
    assert s.reasoning_model == "deepseek-v4-flash"
    assert s.vision_model == "minicpm-v:latest"
    assert s.max_look_rounds == 5
    assert s.sketch_enabled is True
    assert s.cache_enabled is True


def test_env_override(monkeypatch):
    monkeypatch.setenv("DEEPSIGHT_PORT", "9999")
    monkeypatch.setenv("DEEPSIGHT_REASONING_MODEL", "my-model")
    monkeypatch.setenv("DEEPSIGHT_SKETCH_ENABLED", "false")
    s = Settings(_env_file=None)
    assert s.port == 9999
    assert s.reasoning_model == "my-model"
    assert s.sketch_enabled is False


def test_key_properties_none_when_empty():
    s = Settings(_env_file=None)
    assert s.reasoning_key is None
    assert s.vision_key is None


def test_key_properties_when_set(monkeypatch):
    monkeypatch.setenv("DEEPSIGHT_REASONING_API_KEY", "sk-test")
    monkeypatch.setenv("DEEPSIGHT_VISION_API_KEY", "vk-test")
    s = Settings(_env_file=None)
    assert s.reasoning_key == "sk-test"
    assert s.vision_key == "vk-test"
