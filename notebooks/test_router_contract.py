"""Offline contracts for untrusted model classifications and import side effects."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest


@pytest.fixture
def router():
    spec = importlib.util.spec_from_file_location(
        "router_contract_subject", Path(__file__).with_name("resource_aware_optimization.py")
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"openai": SimpleNamespace(OpenAI=Mock())}):
        spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("payload", [[], 0, True, {}, {"classification": "unknown"}, {"classification": []}])
def test_model_classification_has_safe_fallback(router, payload):
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))])
    router.client = Mock()
    router.client.chat.completions.create.return_value = response
    assert router.classify_prompt("question") == {"classification": "simple"}


def test_import_does_not_construct_api_client(router):
    assert router.client is None
    router.OpenAI.assert_not_called()


def test_missing_credentials_fail_before_request(router, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    router.client = None
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        router.classify_prompt("question")
