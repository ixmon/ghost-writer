"""Runtime settings for the GhostWriter CLI and web UI.

One TOML file selects the OpenAI-compatible endpoint (llama.cpp, Ollama,
vLLM, xAI, or anything else that serves /v1/chat/completions), the book
library, and the address the web UI binds.

Lookup for the file, first match wins:

  1. $GHOSTWRITER_CONFIG (if this is set and the file is missing, exit)
  2. ./ghostwriter.toml in the current directory
     (skipped when $GHOSTWRITER_IGNORE_CWD_CONFIG is 1/true/yes/on)
  3. ${XDG_CONFIG_HOME:-~/.config}/ghostwriter/config.toml

Value precedence, highest wins:

  1. CLI flags, applied by the caller on top of load_settings()
  2. Environment variables
  3. The selected file
  4. Built-in defaults (llama.cpp at http://127.0.0.1:8080/v1)

Inside a file, llm.base_url wins over llm.host and llm.port.
GHOSTWRITER_BASE_URL wins over GHOSTWRITER_HOST and GHOSTWRITER_PORT.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping
from urllib.parse import urlsplit, urlunsplit


DEFAULT_LLM_HOST = "127.0.0.1"
DEFAULT_LLM_PORT = 8080
DEFAULT_MAX_TOKENS = 16384
DEFAULT_TEMPERATURE = 0.88
DEFAULT_LIBRARY = "~/ghostwriter"
DEFAULT_WEB_HOST = "0.0.0.0"
DEFAULT_WEB_PORT = 8501

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


@dataclass(frozen=True)
class Settings:
    llm_base_url: str
    llm_host: str
    llm_port: int
    llm_model: str
    llm_api_key: str
    strict_openai: bool
    temperature: float
    max_tokens: int
    library: str
    web_host: str
    web_port: int
    config_path: str | None

    def __repr__(self) -> str:
        key = "set" if self.llm_api_key else "unset"
        return (
            f"Settings(llm_base_url={redact_url(self.llm_base_url)!r}, "
            f"model={self.llm_model!r}, api_key={key}, "
            f"library={self.library!r}, web={self.web_host}:{self.web_port})"
        )

    def with_base_url(self, base_url: str) -> Settings:
        normalized = normalize_base_url(base_url)
        host, port = split_base_url(normalized)
        return replace(self, llm_base_url=normalized, llm_host=host, llm_port=port)

    def retarget(self, base_url: str, env: Mapping[str, str]) -> Settings:
        """Apply a one-run --base-url without carrying another endpoint's key.

        Switching onto *.api.x.ai picks up XAI_API_KEY and strict mode.
        Switching away drops a key that belonged to the previous endpoint,
        unless GHOSTWRITER_API_KEY is set explicitly. An unchanged URL keeps
        the key and strict flag already resolved from the file.
        """
        updated = self.with_base_url(base_url)
        if updated.llm_base_url == self.llm_base_url:
            return updated
        if "GHOSTWRITER_API_KEY" in env:
            api_key = env["GHOSTWRITER_API_KEY"].strip()
        elif _is_xai_host(updated.llm_host):
            api_key = (env.get("XAI_API_KEY") or "").strip()
        else:
            api_key = ""
        env_strict = _present(env, "GHOSTWRITER_STRICT_OPENAI")
        if env_strict is None:
            strict = _is_xai_host(updated.llm_host)
        else:
            strict = _as_bool(env_strict, "GHOSTWRITER_STRICT_OPENAI")
        return replace(updated, llm_api_key=api_key, strict_openai=strict)

    def with_model(self, model: str) -> Settings:
        return replace(self, llm_model=model.strip())

    def with_strict(self, strict: bool) -> Settings:
        return replace(self, strict_openai=strict)

    def uses_configured_endpoint(self, host: str, port: int) -> bool:
        return host.lower() == self.llm_host.lower() and int(port) == int(self.llm_port)

    def chat_url(self, host: str, port: int) -> str:
        if self.uses_configured_endpoint(host, port):
            root = self.llm_base_url
        else:
            root = build_base_url(host, port)
        return root.rstrip("/") + "/chat/completions"

    def request_headers(self, host: str, port: int) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.uses_configured_endpoint(host, port) and self.llm_api_key:
            headers["Authorization"] = f"Bearer {self.llm_api_key}"
        return headers

    def connection(self, host: str | None, port: int | None) -> tuple[str, dict[str, str]]:
        """Resolve a request's host/port to a chat-completions URL and headers.

        A missing host or port means "use the configured endpoint".
        """
        use_host = self.llm_host if not host else host
        use_port = self.llm_port if port is None else int(port)
        return self.chat_url(use_host, use_port), self.request_headers(use_host, use_port)

    def public_dict(self) -> dict:
        """Settings safe to send to the browser. Never includes the API key."""
        return {
            "llm_base_url": redact_url(self.llm_base_url),
            "llm_host": self.llm_host,
            "llm_port": self.llm_port,
            "llm_model": self.llm_model,
            "has_api_key": bool(self.llm_api_key),
            "strict_openai": self.strict_openai,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "library": self.library,
            "web_host": self.web_host,
            "web_port": self.web_port,
            "config_path": self.config_path,
        }


_CURRENT: Settings | None = None


def load_settings() -> Settings:
    """Process-wide settings. The first call reads the environment and disk."""
    global _CURRENT
    if _CURRENT is None:
        _CURRENT = resolve_settings(os.environ, Path.cwd())
    return _CURRENT


def set_settings(settings: Settings) -> Settings:
    global _CURRENT
    _CURRENT = settings
    return settings


def reset_settings() -> None:
    global _CURRENT
    _CURRENT = None


def resolve_settings(env: Mapping[str, str], cwd: Path) -> Settings:
    path = find_config_file(env, cwd)
    data: dict = {}
    if path is not None:
        data = _load_toml(path)

    llm = _section(data, "llm")
    library_table = _section(data, "library")
    web = _section(data, "web")

    host = DEFAULT_LLM_HOST
    port = DEFAULT_LLM_PORT
    base_url: str | None = None

    file_base = _clean(llm.get("base_url"))
    if file_base:
        base_url = normalize_base_url(file_base)
        host, port = split_base_url(base_url)
    else:
        file_host = _clean(llm.get("host"))
        if file_host:
            host = file_host
        if llm.get("port") not in (None, ""):
            port = _as_port(llm.get("port"), "llm.port")

    env_base = _present(env, "GHOSTWRITER_BASE_URL")
    if env_base:
        base_url = normalize_base_url(env_base)
        host, port = split_base_url(base_url)
    elif _present(env, "GHOSTWRITER_HOST") or _present(env, "GHOSTWRITER_PORT"):
        env_host = _present(env, "GHOSTWRITER_HOST")
        if env_host:
            host = env_host
        env_port = _present(env, "GHOSTWRITER_PORT")
        if env_port:
            port = _as_port(env_port, "GHOSTWRITER_PORT")
        base_url = None

    if base_url is None:
        base_url = build_base_url(host, port)

    model = _clean(llm.get("model")) or ""
    env_model = _present(env, "GHOSTWRITER_MODEL")
    if env_model is not None:
        model = env_model

    if "GHOSTWRITER_API_KEY" in env:
        api_key = env["GHOSTWRITER_API_KEY"].strip()
    else:
        api_key = _clean(llm.get("api_key")) or ""
        if not api_key and _is_xai_host(host):
            api_key = _clean(env.get("XAI_API_KEY")) or ""

    strict = False
    strict_explicit = False
    if "strict_openai" in llm:
        strict = _as_bool(llm.get("strict_openai"), "llm.strict_openai")
        strict_explicit = True
    env_strict = _present(env, "GHOSTWRITER_STRICT_OPENAI")
    if env_strict is not None:
        strict = _as_bool(env_strict, "GHOSTWRITER_STRICT_OPENAI")
        strict_explicit = True
    if not strict_explicit and _is_xai_host(host):
        strict = True

    temperature = DEFAULT_TEMPERATURE
    if llm.get("temperature") not in (None, ""):
        temperature = _as_float(llm.get("temperature"), "llm.temperature")
    env_temp = _present(env, "GHOSTWRITER_TEMPERATURE")
    if env_temp is not None:
        temperature = _as_float(env_temp, "GHOSTWRITER_TEMPERATURE")

    max_tokens = DEFAULT_MAX_TOKENS
    if llm.get("max_tokens") not in (None, ""):
        max_tokens = _as_int(llm.get("max_tokens"), "llm.max_tokens")
    env_tokens = _present(env, "GHOSTWRITER_MAX_TOKENS")
    if env_tokens is not None:
        max_tokens = _as_int(env_tokens, "GHOSTWRITER_MAX_TOKENS")

    library = str(Path(DEFAULT_LIBRARY).expanduser())
    file_library = _clean(library_table.get("path"))
    if file_library:
        library = str(Path(file_library).expanduser())
    legacy_library = _present(env, "BOOKWRIGHT_LIBRARY")
    if legacy_library:
        library = str(Path(legacy_library).expanduser())
    env_library = _present(env, "GHOSTWRITER_LIBRARY")
    if env_library:
        library = str(Path(env_library).expanduser())

    web_host = DEFAULT_WEB_HOST
    web_port = DEFAULT_WEB_PORT
    file_web_host = _clean(web.get("host"))
    if file_web_host:
        web_host = file_web_host
    if web.get("port") not in (None, ""):
        web_port = _as_port(web.get("port"), "web.port")
    env_web_host = _present(env, "GHOSTWRITER_WEB_HOST")
    if env_web_host:
        web_host = env_web_host
    env_web_port = _present(env, "GHOSTWRITER_WEB_PORT")
    if env_web_port:
        web_port = _as_port(env_web_port, "GHOSTWRITER_WEB_PORT")

    return Settings(
        llm_base_url=base_url,
        llm_host=host,
        llm_port=port,
        llm_model=model,
        llm_api_key=api_key,
        strict_openai=strict,
        temperature=temperature,
        max_tokens=max_tokens,
        library=library,
        web_host=web_host,
        web_port=web_port,
        config_path=str(path) if path else None,
    )


def find_config_file(env: Mapping[str, str], cwd: Path) -> Path | None:
    explicit = _present(env, "GHOSTWRITER_CONFIG")
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise SystemExit(f"GHOSTWRITER_CONFIG does not point at a file: {path}")
        return path

    ignore_cwd = _as_bool(
        _present(env, "GHOSTWRITER_IGNORE_CWD_CONFIG") or "false",
        "GHOSTWRITER_IGNORE_CWD_CONFIG",
    )
    if not ignore_cwd:
        local = cwd / "ghostwriter.toml"
        if local.is_file():
            return local

    xdg = _present(env, "XDG_CONFIG_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".config"
    user = base / "ghostwriter" / "config.toml"
    if user.is_file():
        return user
    return None


def normalize_base_url(url: str) -> str:
    raw = url.strip().rstrip("/")
    suffix = "/chat/completions"
    if raw.endswith(suffix):
        raw = raw[: -len(suffix)].rstrip("/")
    parts = urlsplit(raw)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise SystemExit(
            "llm.base_url must be an http(s) URL whose path includes /v1, "
            f"got: {url}"
        )
    return raw


def split_base_url(url: str) -> tuple[str, int]:
    parts = urlsplit(url)
    host = parts.hostname
    if not host:
        raise SystemExit(f"llm.base_url is missing a host: {url}")
    if parts.port is not None:
        port = parts.port
    elif parts.scheme == "https":
        port = 443
    else:
        port = 80
    return host, port


def build_base_url(host: str, port: int) -> str:
    host = host.strip()
    if not host:
        raise SystemExit("LLM host is empty")
    literal = f"[{host}]" if ":" in host else host
    return f"http://{literal}:{int(port)}/v1"


def redact_url(url: str) -> str:
    """Drop userinfo so a URL is safe to print or send to the browser."""
    parts = urlsplit(url)
    if parts.username is None and parts.password is None:
        return url
    host = parts.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    netloc = f"{host}:{parts.port}" if parts.port else host
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, ""))


def _is_xai_host(host: str) -> bool:
    host = host.lower()
    return host == "api.x.ai" or host.endswith(".api.x.ai")


def _load_toml(path: Path) -> dict:
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise SystemExit(f"Could not parse {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"{path} must be a TOML table")
    return data


def _section(data: dict, name: str) -> dict:
    raw = data.get(name, {})
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise SystemExit(f"[{name}] in the config must be a table")
    return raw


def _clean(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _present(env: Mapping[str, str], name: str) -> str | None:
    if name not in env:
        return None
    return _clean(env.get(name))


def _as_bool(value, name: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise SystemExit(f"{name} must be true or false, got {value!r}")


def _as_int(value, name: str) -> int:
    if isinstance(value, bool):
        raise SystemExit(f"{name} must be an integer, got {value!r}")
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        raise SystemExit(f"{name} must be an integer, got {value!r}") from None


def _as_port(value, name: str) -> int:
    port = _as_int(value, name)
    if not 1 <= port <= 65535:
        raise SystemExit(f"{name} must be between 1 and 65535, got {port}")
    return port


def _as_float(value, name: str) -> float:
    if isinstance(value, bool):
        raise SystemExit(f"{name} must be a number, got {value!r}")
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        raise SystemExit(f"{name} must be a number, got {value!r}") from None
