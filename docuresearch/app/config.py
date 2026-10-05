"""Runtime settings — implementation plan §8.6.

Settings are read from a TOML file (``config.toml`` in the project directory,
or the path in ``DOCURESEARCH_CONFIG``) and then overridden by environment
variables. ``config.example.toml`` documents every option.

Secrets never go in the TOML file: the LLM API key is read only from
``DOCURESEARCH_LLM_API_KEY``. Environment variables may also be placed in
``.env`` in the project directory (git-ignored); real environment variables
take precedence over it.

Environment overrides:
    DOCURESEARCH_DB_PATH        storage.db_path
    DOCURESEARCH_LLM_BASE_URL   llm.base_url
    DOCURESEARCH_LLM_MODEL      llm.model
    DOCURESEARCH_LLM_TIMEOUT    llm.timeout_seconds
    DOCURESEARCH_LLM_API_KEY    (environment only)
    DOCURESEARCH_ALLOW_REGISTRATION  auth.allow_registration (true/false)
    DOCURESEARCH_COOKIE_SECURE       auth.cookie_secure (true/false)
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_DIR / "config.toml"
DEFAULT_ENV_PATH = PROJECT_DIR / ".env"


class ConfigError(ValueError):
    """Invalid configuration value."""


@dataclass(frozen=True)
class LLMSettings:
    base_url: str | None = None
    model: str | None = None
    api_key: str | None = field(default=None, repr=False)
    timeout_seconds: float = 120.0

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.model)


@dataclass(frozen=True)
class Settings:
    db_path: Path = PROJECT_DIR / "data" / "storage" / "docuresearch.db"
    embedding_model: str = "all-MiniLM-L6-v2"
    max_candidates: int = 20
    # Tuned on the evaluation dataset (2026-10-02): 0.7/0.3 with stopwords gave
    # recall@20 0.930 vs 0.884 and MRR 0.620 vs 0.585 for the old 0.5/0.5.
    semantic_weight: float = 0.7
    keyword_weight: float = 0.3
    max_history_turns: int = 5
    allow_registration: bool = True
    session_lifetime_days: int = 7
    cookie_secure: bool = False  # set True when served over HTTPS
    llm: LLMSettings = field(default_factory=LLMSettings)

    def __post_init__(self) -> None:
        if self.max_candidates < 1:
            raise ConfigError(f"retrieval.max_candidates must be >= 1, got {self.max_candidates}")
        if self.session_lifetime_days < 1:
            raise ConfigError(
                f"auth.session_lifetime_days must be >= 1, got {self.session_lifetime_days}"
            )
        if self.max_history_turns < 1:
            raise ConfigError(
                f"conversation.max_turns must be >= 1, got {self.max_history_turns}"
            )
        for name in ("semantic_weight", "keyword_weight"):
            if getattr(self, name) < 0:
                raise ConfigError(f"retrieval.{name} must be >= 0")
        if self.semantic_weight + self.keyword_weight == 0:
            raise ConfigError("retrieval weights must not both be 0")

    @classmethod
    def load(
        cls,
        path: str | Path | None = None,
        env: dict[str, str] | None = None,
    ) -> Settings:
        """Load settings from TOML (if present) and environment overrides.

        Args:
            path: Config file. Defaults to $DOCURESEARCH_CONFIG, then
                ``config.toml`` in the project directory. A missing default
                file is fine (built-in defaults apply); a missing explicit
                file is an error.
            env: Environment mapping. Defaults to ``os.environ`` layered over
                the project ``.env`` file.
        """
        if env is None:
            env = {**read_env_file(DEFAULT_ENV_PATH), **os.environ}
        explicit = path or env.get("DOCURESEARCH_CONFIG")
        config_path = Path(explicit) if explicit else DEFAULT_CONFIG_PATH
        data: dict[str, Any] = {}
        if config_path.is_file():
            with open(config_path, "rb") as f:
                data = tomllib.load(f)
        elif explicit:
            raise ConfigError(f"Config file not found: {config_path}")

        storage = data.get("storage", {})
        retrieval = data.get("retrieval", {})
        conversation = data.get("conversation", {})
        auth = data.get("auth", {})
        llm = data.get("llm", {})

        db_path = Path(env.get("DOCURESEARCH_DB_PATH") or storage.get("db_path", cls.db_path))
        if not db_path.is_absolute():
            db_path = PROJECT_DIR / db_path

        return cls(
            db_path=db_path,
            embedding_model=retrieval.get("embedding_model", cls.embedding_model),
            max_candidates=int(retrieval.get("max_candidates", cls.max_candidates)),
            semantic_weight=float(retrieval.get("semantic_weight", cls.semantic_weight)),
            keyword_weight=float(retrieval.get("keyword_weight", cls.keyword_weight)),
            max_history_turns=int(conversation.get("max_turns", cls.max_history_turns)),
            allow_registration=_flag(
                env.get("DOCURESEARCH_ALLOW_REGISTRATION"),
                auth.get("allow_registration", cls.allow_registration),
            ),
            session_lifetime_days=int(
                auth.get("session_lifetime_days", cls.session_lifetime_days)
            ),
            cookie_secure=_flag(
                env.get("DOCURESEARCH_COOKIE_SECURE"), auth.get("cookie_secure", cls.cookie_secure)
            ),
            llm=LLMSettings(
                # Trimmed: Docker's --env-file keeps stray spaces (KEY= value),
                # which would otherwise produce an invalid Authorization header.
                base_url=(
                    _clean(env.get("DOCURESEARCH_LLM_BASE_URL")) or _clean(llm.get("base_url"))
                ),
                model=_clean(env.get("DOCURESEARCH_LLM_MODEL")) or _clean(llm.get("model")),
                api_key=_clean(env.get("DOCURESEARCH_LLM_API_KEY")),
                timeout_seconds=float(
                    env.get("DOCURESEARCH_LLM_TIMEOUT")
                    or llm.get("timeout_seconds", LLMSettings.timeout_seconds)
                ),
            ),
        )


def read_env_file(path: Path) -> dict[str, str]:
    """Parse simple ``KEY=value`` lines; blank lines and ``#`` comments are ignored."""
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        values[key.strip().removeprefix("export ").strip()] = value
    return values


def _flag(env_value: str | None, default: object) -> bool:
    """A boolean from an environment string (if set) or a TOML value."""
    if env_value is None or env_value == "":
        return bool(default)
    value = env_value.strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise ConfigError(f"Expected true/false, got {env_value!r}")


def _clean(value: object) -> str | None:
    """A trimmed string, or None if missing or blank."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None
