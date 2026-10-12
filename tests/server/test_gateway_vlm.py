"""VLM imports use the same endpoint semantics and identities as OV backends."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from openviking.config.vlm import AccountVLMProvider
from openviking.server.gateway_vlm import vlm_upstreams
from openviking_cli.utils.config.vlm_config import VLMConfig


def configured(provider, model="model", **kwargs):
    return VLMConfig(
        model=model,
        credentials=[{"id": "primary", "provider": provider, "api_key": "secret", **kwargs}],
    )


@pytest.mark.parametrize(
    "provider,model,base,protocol,vendor",
    [
        ("openai", "gpt", "https://api.openai.com/v1", "chat", "openai"),
        ("volcengine", "ep-123", "https://ark.cn-beijing.volces.com/api/v3", "chat", "ark"),
        ("litellm", "anthropic/claude", "https://api.anthropic.com", "anthropic", "anthropic"),
        ("litellm", "ollama/qwen", "http://localhost:11434/v1", "chat", "generic"),
        ("litellm", "ollama_chat/qwen", "http://localhost:11434/v1", "chat", "generic"),
    ],
)
def test_default_provider_mapping(provider, model, base, protocol, vendor):
    result = vlm_upstreams(configured(provider, model))
    assert result["skipped"] == []
    (upstream,) = result["upstreams"]
    assert (upstream["id"], upstream["base_url"], upstream["protocol"], upstream["vendor"]) == (
        "vlm-primary",
        base,
        protocol,
        vendor,
    )
    assert upstream["models"] == [model.split("/", 1)[-1]]


@pytest.mark.parametrize(
    "path,expected",
    [("/coding/paas/v4", True), ("/plan", True), ("/myplan/v1", False), ("/coding-plan", False)],
)
def test_coding_plan_requires_explicit_consent(path, expected):
    (upstream,) = vlm_upstreams(configured("openai", api_base="https://example.com" + path))[
        "upstreams"
    ]
    assert upstream["coding_plan"] == expected
    assert upstream["allow_coding_plan"] is False


def test_kimi_alias_base_and_custom_user_agent():
    (upstream,) = vlm_upstreams(
        configured("kimi", "kimi-code", extra_headers={"user-agent": "mine"})
    )["upstreams"]
    assert upstream["base_url"] == "https://api.kimi.com/coding/v1"
    assert upstream["headers"] == {"user-agent": "mine"}
    assert upstream["models"] == ["kimi-for-coding"]
    (default,) = vlm_upstreams(configured("kimi"))["upstreams"]
    assert default["headers"]["User-Agent"] == "KimiCLI/1.30.0"


def test_glm_and_byteplus():
    (glm,) = vlm_upstreams(configured("glm"))["upstreams"]
    assert glm["base_url"] == "https://api.z.ai/api/coding/paas/v4"
    assert glm["coding_plan"] is True
    (ark,) = vlm_upstreams(
        configured(
            "volcengine",
            api_base="https://ark.ap-southeast.bytepluses.com/api/v3",
            extra_headers={"X-Client-Request-Id": "ov", "custom": "value"},
        )
    )["upstreams"]
    assert ark["vendor"] == "byteplus"
    assert ark["headers"] == {"custom": "value"}


@pytest.mark.parametrize(
    "provider,model,reason",
    [
        ("azure", "deployment", "azure"),
        ("openai-codex", "gpt", "openai_codex"),
        ("litellm", "some/model", "api_base"),
    ],
)
def test_skipped_credentials_have_actionable_reasons(provider, model, reason):
    result = vlm_upstreams(configured(provider, model))
    assert result["upstreams"] == []
    assert result["skipped"][0]["id"] == "vlm-primary"
    assert reason in result["skipped"][0]["reason"]


def test_all_credentials_and_legacy_config_are_imported():
    config = VLMConfig(model="gpt", provider="openai", api_key="secret")
    assert len(vlm_upstreams(config)["upstreams"]) == 1
    config = VLMConfig(
        model="gpt",
        credentials=[{"id": key, "provider": "openai", "api_key": key} for key in ("a", "b")],
    )
    assert [u["id"] for u in vlm_upstreams(config)["upstreams"]] == ["vlm-a", "vlm-b"]


async def test_effective_configuration_is_detached_and_does_not_create_clients():
    config = configured("openai")
    provider = object.__new__(AccountVLMProvider)
    provider._select = AsyncMock(return_value=SimpleNamespace(config=config))
    effective = await provider.effective_vlm_config("account")
    assert effective is not config
    assert effective.credentials[0].api_key == "secret"
    effective.credentials[0].api_key = "changed"
    assert config.credentials[0].api_key == "secret"
    provider._select.assert_awaited_once_with("account", "vlm")


async def test_effective_configuration_respects_account_credentials_and_cluster_fallback():
    from openviking.config.account_config import AccountVLMConfig

    cluster = configured("openai", api_base="https://cluster.example/v1")
    account = AccountVLMConfig(
        model="account-model",
        credentials=[
            {
                "id": "own",
                "provider": "openai",
                "api_key": "account-secret",
                "api_base": "https://account.example/v1",
            }
        ],
    )
    provider = AccountVLMProvider(SimpleNamespace(add_update_consumer=lambda **kwargs: None))
    provider._view = AsyncMock(
        return_value=SimpleNamespace(
            account=SimpleNamespace(vlm=account), cluster=SimpleNamespace(vlm=cluster)
        )
    )
    effective = await provider.effective_vlm_config("a")
    assert effective.model == "account-model"
    assert effective.credentials[0].api_key == "account-secret"
    assert effective.credentials[0].api_base == "https://account.example/v1"
    provider._view.return_value.account.vlm = None
    fallback = await provider.effective_vlm_config("b")
    assert fallback.credentials[0].api_key == "secret"
    assert fallback.credentials[0].api_base == "https://cluster.example/v1"
