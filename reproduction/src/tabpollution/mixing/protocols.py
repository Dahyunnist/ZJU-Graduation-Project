"""P1-P5 table/generator/domain and record-isolation validators."""

from __future__ import annotations

from typing import Any


class ProtocolError(ValueError):
    pass


def _sets(manifest: dict[str, Any], prefix: str, field: str) -> set[str]:
    return set(str(item) for item in manifest[f"{prefix}_{field}"])


def validate_protocol(protocol: str, manifest: dict[str, Any]) -> dict[str, Any]:
    train_tables = _sets(manifest, "train", "tables")
    test_tables = _sets(manifest, "test", "tables")
    train_generators = _sets(manifest, "train", "generators")
    test_generators = _sets(manifest, "test", "generators")
    train_records = _sets(manifest, "train", "records")
    test_records = _sets(manifest, "test", "records")
    if train_records & test_records:
        raise ProtocolError(f"Record leakage: {sorted(train_records & test_records)[:5]}")
    if protocol == "P1":
        if train_tables != test_tables or train_generators != test_generators:
            raise ProtocolError("P1 requires the same tables and generators")
    elif protocol == "P2":
        if train_tables != test_tables or train_generators & test_generators:
            raise ProtocolError("P2 requires same tables and disjoint generators")
    elif protocol == "P3":
        if train_tables & test_tables:
            raise ProtocolError(f"P3 table leakage: {sorted(train_tables & test_tables)}")
    elif protocol == "P4":
        if train_tables & test_tables or train_generators & test_generators:
            raise ProtocolError("P4 requires disjoint tables and generators")
    elif protocol == "P5":
        train_domains = _sets(manifest, "train", "domains")
        test_domains = _sets(manifest, "test", "domains")
        if train_domains & test_domains:
            raise ProtocolError(f"P5 domain leakage: {sorted(train_domains & test_domains)}")
    else:
        raise ProtocolError(f"Unknown protocol: {protocol}")
    return {
        "protocol": protocol,
        "passed": True,
        "include_in_p3_macro": protocol != "P5",
    }

