import pytest

from ramanujan.config import Settings
from ramanujan.llm.factory import build_llm
from ramanujan.llm.openai_compat import OpenAICompatClient

FAST = Settings(min_seconds_between_llm_calls=0)

ALL_KEY_ENVS = [
    "GEMINI_API_KEY", "OPENROUTER_API_KEY", "OPENCODE_API_KEY", "OPENCODE_ZEN_API_KEY",
    "RAMANUJAN_API_KEY", "RAMANUJAN_BASE_URL", "RAMANUJAN_MODEL", "RAMANUJAN_PROVIDER",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    # a developer's real .env must not leak keys back into these tests
    monkeypatch.setattr("ramanujan.llm.factory.load_dotenv", lambda *a, **k: None)
    for env in ALL_KEY_ENVS:
        monkeypatch.delenv(env, raising=False)


def test_no_keys_gives_actionable_error():
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        build_llm(settings=FAST)


def test_autodetect_openrouter(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    llm = build_llm(settings=FAST)
    assert isinstance(llm, OpenAICompatClient)
    assert llm.base_url == "https://openrouter.ai/api/v1"
    assert llm.api_key == "or-key"
    assert llm.model  # a tool-calling-capable default is preset


def test_autodetect_opencode_zen_and_aliases(monkeypatch):
    monkeypatch.setenv("OPENCODE_ZEN_API_KEY", "zen-key")
    for provider in (None, "opencode", "opencode-zen", "zen"):
        llm = build_llm(provider, settings=FAST)
        assert isinstance(llm, OpenAICompatClient)
        assert llm.base_url == "https://opencode.ai/zen/v1"
        assert llm.api_key == "zen-key"


def test_model_and_base_url_overrides(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.setenv("RAMANUJAN_MODEL", "qwen/qwen3-coder")
    monkeypatch.setenv("RAMANUJAN_BASE_URL", "https://proxy.example/v1")
    llm = build_llm("openrouter", settings=FAST)
    assert llm.model == "qwen/qwen3-coder"
    assert llm.base_url == "https://proxy.example/v1"


def test_custom_provider(monkeypatch):
    monkeypatch.setenv("RAMANUJAN_API_KEY", "local")
    monkeypatch.setenv("RAMANUJAN_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("RAMANUJAN_MODEL", "qwen2.5-coder:32b")
    llm = build_llm(settings=FAST)  # auto-detected as custom
    assert isinstance(llm, OpenAICompatClient)
    assert llm.base_url == "http://localhost:11434/v1"


def test_provider_selected_without_its_key_errors(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        build_llm("openrouter", settings=FAST)


def test_unknown_provider_errors():
    with pytest.raises(RuntimeError, match="Unknown LLM provider"):
        build_llm("wat", settings=FAST)
