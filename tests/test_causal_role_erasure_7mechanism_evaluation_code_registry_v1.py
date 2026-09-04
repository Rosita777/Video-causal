from pathlib import Path

import pytest

from scripts import causal_role_erasure_7mechanism_evaluation_code_registry_v1 as registry


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_registry_binds_exactly_five_current_evaluation_scripts():
    value = registry.build_registry(PROJECT_ROOT)

    assert value["protocol"] == registry.PROTOCOL
    assert [row["path"] for row in value["files"]] == list(registry.SCRIPT_PATHS)
    assert len(value["files"]) == 5
    assert registry.validate_registry(value, PROJECT_ROOT) == value


def test_registry_rejects_a_changed_script_digest():
    value = registry.build_registry(PROJECT_ROOT)
    value["files"][0]["sha256"] = "0" * 64

    with pytest.raises(ValueError, match="self-commitment|differ"):
        registry.validate_registry(value, PROJECT_ROOT)
