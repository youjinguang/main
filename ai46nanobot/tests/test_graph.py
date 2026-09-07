from __future__ import annotations

from unittest.mock import patch

from mini_nanobot.config import AppConfig, ProviderConfig
from mini_nanobot.graph import build_llm


def _make_cfg() -> AppConfig:
    return AppConfig(
        provider=ProviderConfig(
            api_key="sk-test",
            api_base="https://api.example.com/v1",
            model="gpt-4o-mini",
            temperature=0.3,
            max_tokens=1024,
            timeout_seconds=30,
        )
    )


def test_build_llm_passes_provider_config() -> None:
    cfg = _make_cfg()
    fake_llm = object()

    with patch("mini_nanobot.graph.ChatOpenAI", return_value=fake_llm) as mock_llm:
        llm = build_llm(cfg)

    assert llm is fake_llm
    mock_llm.assert_called_once_with(
        api_key="sk-test",
        base_url="https://api.example.com/v1",
        model="gpt-4o-mini",
        temperature=0.3,
        max_tokens=1024,
        timeout=30,
    )
