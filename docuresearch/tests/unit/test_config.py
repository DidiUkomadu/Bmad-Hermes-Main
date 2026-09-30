"""Tests for runtime settings (implementation plan §8.6)."""

from __future__ import annotations

import pytest

from app.config import PROJECT_DIR, ConfigError, Settings

EXAMPLE = PROJECT_DIR / "config.example.toml"


def test_defaults_without_config_file(tmp_path, monkeypatch):
    monkeypatch.setattr("app.config.DEFAULT_CONFIG_PATH", tmp_path / "absent.toml")
    s = Settings.load(env={})
    assert s.max_candidates == 20
    assert s.max_history_turns == 5
    assert (s.semantic_weight, s.keyword_weight) == (0.5, 0.5)
    assert s.llm.configured is False


def test_example_config_loads(tmp_path):
    s = Settings.load(EXAMPLE, env={})
    assert s.db_path == PROJECT_DIR / "data" / "storage" / "docuresearch.db"
    assert s.llm.base_url == "https://inference-api.nousresearch.com/v1"
    assert s.llm.configured is False  # model left blank in the example


def test_toml_values_and_env_overrides(tmp_path):
    cfg = tmp_path / "c.toml"
    cfg.write_text(
        '[storage]\ndb_path = "/tmp/x.db"\n'
        "[retrieval]\nmax_candidates = 8\nsemantic_weight = 0.7\nkeyword_weight = 0.3\n"
        "[conversation]\nmax_turns = 3\n"
        '[llm]\nbase_url = "http://toml/v1"\nmodel = "toml-model"\n'
    )
    s = Settings.load(cfg, env={
        "DOCURESEARCH_LLM_MODEL": "env-model",
        "DOCURESEARCH_LLM_API_KEY": "secret",
        "DOCURESEARCH_DB_PATH": "data/other.db",
    })
    assert s.max_candidates == 8
    assert (s.semantic_weight, s.keyword_weight) == (0.7, 0.3)
    assert s.max_history_turns == 3
    assert s.llm.base_url == "http://toml/v1"
    assert s.llm.model == "env-model"
    assert s.llm.api_key == "secret"
    assert s.llm.configured is True
    assert s.db_path == PROJECT_DIR / "data" / "other.db"  # relative → project dir


def test_api_key_is_never_read_from_toml(tmp_path):
    cfg = tmp_path / "c.toml"
    cfg.write_text('[llm]\napi_key = "leaked"\n')
    assert Settings.load(cfg, env={}).llm.api_key is None


def test_api_key_not_in_repr():
    s = Settings.load(EXAMPLE, env={"DOCURESEARCH_LLM_API_KEY": "secret"})
    assert "secret" not in repr(s)


def test_config_path_from_env(tmp_path):
    cfg = tmp_path / "c.toml"
    cfg.write_text("[retrieval]\nmax_candidates = 4\n")
    assert Settings.load(env={"DOCURESEARCH_CONFIG": str(cfg)}).max_candidates == 4


def test_missing_explicit_config_file_is_an_error(tmp_path):
    with pytest.raises(ConfigError):
        Settings.load(tmp_path / "missing.toml", env={})


@pytest.mark.parametrize("toml", [
    "[retrieval]\nmax_candidates = 0\n",
    "[conversation]\nmax_turns = 0\n",
    "[retrieval]\nsemantic_weight = -1\n",
    "[retrieval]\nsemantic_weight = 0\nkeyword_weight = 0\n",
])
def test_invalid_values_rejected(tmp_path, toml):
    cfg = tmp_path / "c.toml"
    cfg.write_text(toml)
    with pytest.raises(ConfigError):
        Settings.load(cfg, env={})


def test_env_file_is_read_and_real_env_wins(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\n"
        "DOCURESEARCH_LLM_BASE_URL=http://from-file/v1\n"
        'export DOCURESEARCH_LLM_MODEL="file-model"\n'
        "DOCURESEARCH_LLM_API_KEY='file-key'\n"
        "\n"
    )
    monkeypatch.setattr("app.config.DEFAULT_ENV_PATH", env_file)
    monkeypatch.setattr("app.config.DEFAULT_CONFIG_PATH", tmp_path / "absent.toml")
    monkeypatch.setenv("DOCURESEARCH_LLM_MODEL", "real-env-model")
    monkeypatch.delenv("DOCURESEARCH_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("DOCURESEARCH_LLM_API_KEY", raising=False)

    s = Settings.load()
    assert s.llm.base_url == "http://from-file/v1"
    assert s.llm.model == "real-env-model"
    assert s.llm.api_key == "file-key"


def test_empty_env_file_value_means_unset(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("DOCURESEARCH_LLM_API_KEY=\n")
    monkeypatch.setattr("app.config.DEFAULT_ENV_PATH", env_file)
    monkeypatch.delenv("DOCURESEARCH_LLM_API_KEY", raising=False)
    assert Settings.load().llm.api_key is None
