from __future__ import annotations

import pytest

from tabpollution.mixing.protocols import ProtocolError, validate_protocol


def manifest(protocol: str) -> dict:
    base = {
        "train_tables": ["a"], "test_tables": ["a"],
        "train_generators": ["g1"], "test_generators": ["g1"],
        "train_records": ["r1"], "test_records": ["r2"],
        "train_domains": ["social"], "test_domains": ["social"],
    }
    if protocol == "P2":
        base["test_generators"] = ["g2"]
    elif protocol == "P3":
        base["test_tables"] = ["b"]
    elif protocol == "P4":
        base["test_tables"], base["test_generators"] = ["b"], ["g2"]
    elif protocol == "P5":
        base["test_tables"], base["test_domains"] = ["b"], ["finance"]
    return base


@pytest.mark.parametrize("protocol", ["P1", "P2", "P3", "P4", "P5"])
def test_protocol_positive_examples(protocol: str) -> None:
    assert validate_protocol(protocol, manifest(protocol))["passed"]


def test_p5_is_excluded_from_p3_macro() -> None:
    assert not validate_protocol("P5", manifest("P5"))["include_in_p3_macro"]


@pytest.mark.parametrize("protocol", ["P1", "P2", "P3", "P4", "P5"])
def test_record_leakage_is_rejected_for_every_protocol(protocol: str) -> None:
    bad = manifest(protocol)
    bad["test_records"] = ["r1"]
    with pytest.raises(ProtocolError, match="Record leakage"):
        validate_protocol(protocol, bad)


def test_p3_table_leakage_is_rejected() -> None:
    bad = manifest("P3")
    bad["test_tables"] = ["a"]
    with pytest.raises(ProtocolError, match="table leakage"):
        validate_protocol("P3", bad)


def test_p4_generator_leakage_is_rejected() -> None:
    bad = manifest("P4")
    bad["test_generators"] = ["g1"]
    with pytest.raises(ProtocolError, match="disjoint"):
        validate_protocol("P4", bad)


def test_p5_domain_leakage_is_rejected() -> None:
    bad = manifest("P5")
    bad["test_domains"] = ["social"]
    with pytest.raises(ProtocolError, match="domain leakage"):
        validate_protocol("P5", bad)

