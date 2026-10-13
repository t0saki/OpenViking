"""Configuration split and durable encryption-key behavior."""

import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from cryptography.fernet import Fernet
from pydantic import ValidationError

from openviking_gateway.cli import load_config
from openviking_gateway.config import (
    PUBLIC_OPENVIKING_URLS,
    EmbeddedGatewayConfig,
    GatewayConfig,
    StandaloneGatewayConfig,
)


def test_embedded_ignores_removed_fields_with_warning(caplog):
    legacy = {"enabled": True, "url": "http://127.0.0.1:1935", "port": 1935, "workers": 2}
    config = EmbeddedGatewayConfig.model_validate(legacy)
    assert config.enabled
    assert "Ignoring removed gateway settings port, url, workers" in caplog.text
    with pytest.raises(ValidationError):
        EmbeddedGatewayConfig.model_validate({"enabled": True, "urll": "typo"})


def test_standalone_requires_own_url_and_rejects_server_config():
    with pytest.raises(ValidationError):
        StandaloneGatewayConfig()
    with pytest.raises(ValidationError):
        StandaloneGatewayConfig(openviking_url="http://localhost:1933", enabled=True)


def test_key_file_is_atomic_and_survives_restart(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENVIKING_GATEWAY_ENCRYPTION_KEY", raising=False)
    config = GatewayConfig(storage_path=str(tmp_path))
    with ThreadPoolExecutor(max_workers=16) as pool:
        keys = list(pool.map(lambda _: config.encryption_key(), range(32)))
    assert len(set(keys)) == 1
    assert (tmp_path / "encryption.key").stat().st_mode & 0o777 == 0o600
    assert list(tmp_path.glob(".encryption-*")) == []
    (tmp_path / "management.sqlite3").touch()
    assert GatewayConfig(storage_path=str(tmp_path)).encryption_key() == keys[0]


@pytest.mark.parametrize("filename", ["management.sqlite3", "kernel.sqlite3"])
def test_missing_key_with_database_refuses_startup(tmp_path, monkeypatch, filename):
    monkeypatch.delenv("OPENVIKING_GATEWAY_ENCRYPTION_KEY", raising=False)
    (tmp_path / filename).touch()
    with pytest.raises(ValueError, match="original"):
        GatewayConfig(storage_path=str(tmp_path)).encryption_key()
    assert not (tmp_path / "encryption.key").exists()


def test_encryption_environment_takes_precedence(tmp_path, monkeypatch):
    (tmp_path / "encryption.key").write_text("invalid-file-key")
    key = Fernet.generate_key().decode()
    monkeypatch.setenv("MY_GATEWAY_KEY", key)
    assert (
        GatewayConfig(
            storage_path=str(tmp_path), encryption_key_env="MY_GATEWAY_KEY"
        ).encryption_key()
        == key
    )


def test_management_key_expansion_and_length(monkeypatch):
    monkeypatch.setenv("MANAGEMENT_SECRET", "ovgm_" + "a" * 32)
    assert (
        EmbeddedGatewayConfig(management_key="$MANAGEMENT_SECRET").management_key
        == "ovgm_" + "a" * 32
    )
    with pytest.raises(ValidationError, match="at least 32"):
        GatewayConfig(management_key="short")
    with pytest.raises(ValidationError, match="at least 32"):
        GatewayConfig(management_key="$UNDEFINED_MANAGEMENT_SECRET_THAT_IS_LONG")


def test_standalone_config_file_and_env_precedence(tmp_path, monkeypatch):
    for field in StandaloneGatewayConfig.model_fields:
        monkeypatch.delenv("OPENVIKING_GATEWAY_" + field.upper(), raising=False)
    monkeypatch.setenv("OPENVIKING_CONFIG_FILE", str(tmp_path / "ignored-ov.conf"))
    (tmp_path / "ignored-ov.conf").write_text('{"gateway":{"enabled":true}}')
    config_file = tmp_path / "gateway.conf"
    config_file.write_text(json.dumps({"openviking_url": "https://example.test/ov/", "port": 9001}))
    monkeypatch.setenv("OPENVIKING_GATEWAY_CONFIG_FILE", str(config_file))
    monkeypatch.setenv("OPENVIKING_GATEWAY_PORT", "9002")
    monkeypatch.setenv("OPENVIKING_GATEWAY_MANAGEMENT_KEY", "m" * 32)
    config = load_config()
    assert config.openviking_url == "https://example.test/ov"
    assert config.port == 9002
    assert config.management_key == "m" * 32


def test_empty_environment_values_keep_file_values(tmp_path, monkeypatch):
    for field in StandaloneGatewayConfig.model_fields:
        monkeypatch.setenv("OPENVIKING_GATEWAY_" + field.upper(), "")
    config_file = tmp_path / "gateway.conf"
    config_file.write_text(
        json.dumps({"openviking_url": "https://example.test", "management_key": "f" * 32})
    )
    monkeypatch.setenv("OPENVIKING_GATEWAY_CONFIG_FILE", str(config_file))
    config = load_config()
    assert config.openviking_url == "https://example.test"
    assert config.management_key == "f" * 32


def test_public_url_normalization():
    config = StandaloneGatewayConfig(
        openviking_url="https://API.VIKINGDB.CN-BEIJING.VOLCES.COM/openviking/"
    )
    assert config.openviking_url in PUBLIC_OPENVIKING_URLS
    ipv6 = StandaloneGatewayConfig(openviking_url="http://[::1]:1933/")
    assert ipv6.openviking_url == "http://[::1]:1933"
    with pytest.raises(ValidationError):
        StandaloneGatewayConfig(openviking_url="http://localhost:99999")
