"""LOCO partitioning that keeps held-class records outside P27 fitting paths."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class LocoInventory:
    """Disjoint record views for one held VisA class."""

    held_class: str
    fit_rows: tuple[Mapping[str, Any], ...]
    held_rows: tuple[Mapping[str, Any], ...]

    def assert_firewall(self) -> None:
        if not self.fit_rows or not self.held_rows:
            raise RuntimeError("LOCO inventory requires non-empty fit and held partitions")
        if any(row.get("class_name") == self.held_class for row in self.fit_rows):
            raise RuntimeError("held class reached the source fit partition")
        if any(row.get("class_name") != self.held_class for row in self.held_rows):
            raise RuntimeError("non-held class reached the held partition")


def loco_inventory(rows: Iterable[Mapping[str, Any]], held_class: str) -> LocoInventory:
    """Partition metadata records without touching image or mask paths."""
    if not isinstance(held_class, str) or not held_class:
        raise ValueError("held_class must be a non-empty string")
    records = tuple(rows)
    if any(not isinstance(row.get("class_name"), str) for row in records):
        raise ValueError("every record must have a string class_name")
    inventory = LocoInventory(
        held_class=held_class,
        fit_rows=tuple(row for row in records if row["class_name"] != held_class),
        held_rows=tuple(row for row in records if row["class_name"] == held_class),
    )
    inventory.assert_firewall()
    return inventory
