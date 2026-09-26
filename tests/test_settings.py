"""Precedence and endpoint resolution for settings.py."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from settings import (  # noqa: E402
    build_base_url,
    redact_url,
    resolve_settings,
)


class SettingsTests(unittest.TestCase):
    def setUp(self):
        # Isolate every test from the real home directory config.
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.tmp = Path(self._td.name)
        self.xdg = self.tmp / "xdg"
        self.cwd = self.tmp / "cwd"
        self.xdg.mkdir()
        self.cwd.mkdir()
        self.env = {"XDG_CONFIG_HOME": str(self.xdg)}

    def _write(self, directory: Path, text: str, name: str = "config.toml") -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        if name == "config.toml":
            path = directory / "ghostwriter" / "config.toml"
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path = directory / name
        path.write_text(text)
        return path

    def test_defaults_point_at_llama_cpp_and_bind_all_interfaces(self):
        settings = resolve_settings(self.env, self.cwd)
        self.assertEqual(settings.llm_base_url, "http://127.0.0.1:8080/v1")
        self.assertEqual(settings.llm_host, "127.0.0.1")
        self.assertEqual(settings.llm_port, 8080)
        self.assertEqual(settings.llm_model, "")
        self.assertEqual(settings.llm_api_key, "")
        self.assertFalse(settings.strict_openai)
        self.assertEqual(settings.web_host, "0.0.0.0")
        self.assertEqual(settings.web_port, 8501)
        self.assertEqual(settings.library, str(Path("~/ghostwriter").expanduser()))
        self.assertIsNone(settings.config_path)
        self.assertEqual(settings.max_tokens, 16384)
        self.assertAlmostEqual(settings.temperature, 0.88)

    def test_cwd_file_beats_xdg_and_base_url_wins_inside_the_file(self):
        self._write(self.xdg, '[llm]\nbase_url = "http://10.0.0.2:9/v1"\nmodel = "from-xdg"\n')
        local = self.cwd / "ghostwriter.toml"
        local.write_text(
            "\n".join([
                "[llm]",
                'base_url = "https://llm.example:8443/v1/"',
                'host = "should-not-win"',
                "port = 1",
                'model = "from-cwd"',
                "temperature = 0.5",
                "max_tokens = 1000",
                "strict_openai = true",
                'api_key = "file-key"',
                "",
                "[library]",
                'path = "~/books"',
                "",
                "[web]",
                'host = "127.0.0.1"',
                "port = 9000",
                "",
            ])
        )
        settings = resolve_settings(self.env, self.cwd)
        self.assertEqual(settings.config_path, str(local))
        self.assertEqual(settings.llm_base_url, "https://llm.example:8443/v1")
        self.assertEqual(settings.llm_host, "llm.example")
        self.assertEqual(settings.llm_port, 8443)
        self.assertEqual(settings.llm_model, "from-cwd")
        self.assertEqual(settings.llm_api_key, "file-key")
        self.assertTrue(settings.strict_openai)
        self.assertEqual(settings.library, str(Path("~/books").expanduser()))
        self.assertEqual(settings.web_host, "127.0.0.1")
        self.assertEqual(settings.web_port, 9000)
        self.assertEqual(settings.max_tokens, 1000)

    def test_explicit_config_beats_cwd_and_errors_when_missing(self):
        (self.cwd / "ghostwriter.toml").write_text('[llm]\nmodel = "local"\n')
        chosen = self.tmp / "chosen.toml"
        chosen.write_text('[llm]\nbase_url = "http://10.1.1.1:8080/v1"\nmodel = "chosen"\n')
        env = {**self.env, "GHOSTWRITER_CONFIG": str(chosen)}
        settings = resolve_settings(env, self.cwd)
        self.assertEqual(settings.llm_model, "chosen")
        self.assertEqual(settings.llm_host, "10.1.1.1")

        missing = {**self.env, "GHOSTWRITER_CONFIG": str(self.tmp / "nope.toml")}
        with self.assertRaises(SystemExit):
            resolve_settings(missing, self.cwd)

    def test_ignore_cwd_falls_through_to_xdg(self):
        (self.cwd / "ghostwriter.toml").write_text('[llm]\nmodel = "local"\n')
        self._write(self.xdg, '[llm]\nmodel = "xdg"\nbase_url = "http://10.9.9.9:8080/v1"\n')
        env = {**self.env, "GHOSTWRITER_IGNORE_CWD_CONFIG": "1"}
        settings = resolve_settings(env, self.cwd)
        self.assertEqual(settings.llm_model, "xdg")
        self.assertEqual(settings.llm_host, "10.9.9.9")

    def test_env_overrides_file(self):
        (self.cwd / "ghostwriter.toml").write_text(
            "\n".join([
                "[llm]",
                'base_url = "https://llm.example/v1"',
                'model = "file-model"',
                'api_key = "file-key"',
                "strict_openai = false",
                "temperature = 0.2",
                "max_tokens = 50",
                "",
                "[library]",
                'path = "/from/file"',
                "",
                "[web]",
                'host = "127.0.0.1"',
                "port = 1",
                "",
            ])
        )
        env = {
            **self.env,
            "GHOSTWRITER_BASE_URL": "http://127.0.0.1:11434/v1/chat/completions",
            "GHOSTWRITER_MODEL": "qwen2.5:32b",
            "GHOSTWRITER_API_KEY": "env-key",
            "GHOSTWRITER_STRICT_OPENAI": "yes",
            "GHOSTWRITER_TEMPERATURE": "0.4",
            "GHOSTWRITER_MAX_TOKENS": "2048",
            "GHOSTWRITER_LIBRARY": "/from/env",
            "BOOKWRIGHT_LIBRARY": "/legacy",
            "GHOSTWRITER_WEB_HOST": "0.0.0.0",
            "GHOSTWRITER_WEB_PORT": "8502",
        }
        settings = resolve_settings(env, self.cwd)
        self.assertEqual(settings.llm_base_url, "http://127.0.0.1:11434/v1")
        self.assertEqual(settings.llm_port, 11434)
        self.assertEqual(settings.llm_model, "qwen2.5:32b")
        self.assertEqual(settings.llm_api_key, "env-key")
        self.assertTrue(settings.strict_openai)
        self.assertAlmostEqual(settings.temperature, 0.4)
        self.assertEqual(settings.max_tokens, 2048)
        self.assertEqual(settings.library, "/from/env")
        self.assertEqual(settings.web_host, "0.0.0.0")
        self.assertEqual(settings.web_port, 8502)

    def test_host_env_rebuilds_a_local_url(self):
        (self.cwd / "ghostwriter.toml").write_text(
            '[llm]\nbase_url = "https://llm.example/v1"\n'
        )
        env = {**self.env, "GHOSTWRITER_PORT": "11434"}
        settings = resolve_settings(env, self.cwd)
        self.assertEqual(settings.llm_base_url, "http://llm.example:11434/v1")

    def test_library_env_legacy_name_and_blank_model(self):
        (self.cwd / "ghostwriter.toml").write_text("[library]\npath = \"/from/file\"\n")
        env = {**self.env, "BOOKWRIGHT_LIBRARY": "~/old-library"}
        settings = resolve_settings(env, self.cwd)
        self.assertEqual(settings.library, str(Path("~/old-library").expanduser()))

    def test_xai_uses_xai_api_key_and_strict_mode(self):
        env = {
            **self.env,
            "GHOSTWRITER_BASE_URL": "https://api.x.ai/v1",
            "XAI_API_KEY": "xai-secret",
        }
        settings = resolve_settings(env, self.cwd)
        self.assertEqual(settings.llm_host, "api.x.ai")
        self.assertEqual(settings.llm_port, 443)
        self.assertEqual(settings.llm_api_key, "xai-secret")
        self.assertTrue(settings.strict_openai)
        self.assertEqual(
            settings.chat_url("api.x.ai", 443),
            "https://api.x.ai/v1/chat/completions",
        )
        self.assertEqual(
            settings.request_headers("api.x.ai", 443)["Authorization"],
            "Bearer xai-secret",
        )
        # A different host must not receive this key.
        self.assertNotIn("Authorization", settings.request_headers("127.0.0.1", 8080))
        dumped = json.dumps(settings.public_dict())
        self.assertNotIn("xai-secret", dumped)
        self.assertNotIn("llm_api_key", settings.public_dict())
        self.assertTrue(settings.public_dict()["has_api_key"])
        self.assertNotIn("xai-secret", repr(settings))

    def test_explicit_key_and_strict_flag_win_on_xai(self):
        (self.cwd / "ghostwriter.toml").write_text(
            "\n".join([
                "[llm]",
                'base_url = "https://us.api.x.ai/v1"',
                "strict_openai = false",
                'api_key = "from-file"',
                "",
            ])
        )
        env = {**self.env, "XAI_API_KEY": "from-env-xai", "GHOSTWRITER_API_KEY": "explicit"}
        settings = resolve_settings(env, self.cwd)
        self.assertEqual(settings.llm_api_key, "explicit")
        self.assertFalse(settings.strict_openai)
        self.assertEqual(settings.llm_host, "us.api.x.ai")

    def test_xai_key_is_not_sent_to_localhost(self):
        env = {**self.env, "XAI_API_KEY": "xai-secret"}
        settings = resolve_settings(env, self.cwd)
        self.assertEqual(settings.llm_api_key, "")
        self.assertFalse(settings.strict_openai)

    def test_empty_api_key_env_clears_a_file_key(self):
        (self.cwd / "ghostwriter.toml").write_text('[llm]\napi_key = "file-key"\n')
        settings = resolve_settings({**self.env, "GHOSTWRITER_API_KEY": ""}, self.cwd)
        self.assertEqual(settings.llm_api_key, "")

    def test_ipv6_and_override_url(self):
        (self.cwd / "ghostwriter.toml").write_text('[llm]\nbase_url = "http://[::1]:8080/v1"\n')
        settings = resolve_settings(self.env, self.cwd)
        self.assertEqual(settings.llm_host, "::1")
        self.assertEqual(settings.llm_port, 8080)
        self.assertEqual(settings.chat_url("::1", 8080), "http://[::1]:8080/v1/chat/completions")
        self.assertEqual(
            settings.chat_url("::1", 11434),
            "http://[::1]:11434/v1/chat/completions",
        )
        moved = settings.with_base_url("http://10.0.0.8:11434/v1/chat/completions")
        self.assertEqual(moved.llm_base_url, "http://10.0.0.8:11434/v1")
        self.assertEqual(moved.llm_port, 11434)
        url, headers = moved.connection(None, None)
        self.assertEqual(url, "http://10.0.0.8:11434/v1/chat/completions")
        self.assertEqual(headers, {"Content-Type": "application/json"})

    def test_case_insensitive_host_match(self):
        settings = resolve_settings(
            {**self.env, "GHOSTWRITER_BASE_URL": "https://API.X.AI/v1", "XAI_API_KEY": "k"},
            self.cwd,
        )
        self.assertTrue(settings.uses_configured_endpoint("api.x.ai", 443))
        self.assertIn("Authorization", settings.request_headers("API.X.AI", 443))

    def test_bad_url_and_bad_toml(self):
        with self.assertRaises(SystemExit):
            resolve_settings({**self.env, "GHOSTWRITER_BASE_URL": "ftp://nope"}, self.cwd)
        (self.cwd / "ghostwriter.toml").write_text("llm = [\n")
        with self.assertRaises(SystemExit):
            resolve_settings(self.env, self.cwd)

    def test_retarget_switches_endpoint_without_carrying_the_old_key(self):
        settings = resolve_settings(self.env, self.cwd)
        moved = settings.retarget("https://api.x.ai/v1", {"XAI_API_KEY": "sekret"})
        self.assertEqual(moved.llm_base_url, "https://api.x.ai/v1")
        self.assertEqual(moved.llm_api_key, "sekret")
        self.assertTrue(moved.strict_openai)
        self.assertEqual(moved.chat_url("api.x.ai", 443), "https://api.x.ai/v1/chat/completions")

        back = moved.retarget("http://127.0.0.1:8080/v1", {"XAI_API_KEY": "sekret"})
        self.assertEqual(back.llm_base_url, "http://127.0.0.1:8080/v1")
        self.assertEqual(back.llm_api_key, "")
        self.assertFalse(back.strict_openai)

        explicit = moved.retarget(
            "http://127.0.0.1:11434/v1",
            {"GHOSTWRITER_API_KEY": "keep-me", "GHOSTWRITER_STRICT_OPENAI": "true"},
        )
        self.assertEqual(explicit.llm_api_key, "keep-me")
        self.assertTrue(explicit.strict_openai)

        same = moved.retarget("https://api.x.ai/v1", {})
        self.assertEqual(same.llm_api_key, "sekret")

    def test_redact_and_build(self):
        self.assertEqual(
            redact_url("https://user:secret@api.x.ai/v1"),
            "https://api.x.ai/v1",
        )
        self.assertEqual(build_base_url("::1", 8080), "http://[::1]:8080/v1")


if __name__ == "__main__":
    unittest.main()
