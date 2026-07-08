import pytest

from graphon.entities.base_node_data import RetryConfig


def test_first_token_timeout_defaults_to_zero() -> None:
    config = RetryConfig()

    assert config.first_token_timeout == 0
    assert config.first_token_timeout_seconds is None


def test_first_token_timeout_seconds_converts_milliseconds() -> None:
    config = RetryConfig(first_token_timeout=5000)

    assert config.first_token_timeout_seconds == pytest.approx(5.0)


def test_first_token_timeout_seconds_is_none_when_not_positive() -> None:
    assert RetryConfig(first_token_timeout=0).first_token_timeout_seconds is None
    assert RetryConfig(first_token_timeout=-1).first_token_timeout_seconds is None


def test_missing_first_token_timeout_is_backward_compatible() -> None:
    # Workflows serialized before this field existed omit it entirely.
    config = RetryConfig.model_validate({
        "max_retries": 3,
        "retry_interval": 1000,
        "retry_enabled": True,
    })

    assert config.first_token_timeout == 0
    assert config.first_token_timeout_seconds is None


def test_retry_config_round_trips_first_token_timeout() -> None:
    config = RetryConfig(first_token_timeout=1500)
    dumped = config.model_dump(mode="json")

    assert dumped["first_token_timeout"] == 1500
    assert RetryConfig.model_validate(dumped).first_token_timeout == 1500
