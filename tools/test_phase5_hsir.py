#!/usr/bin/env python3
"""Deterministic foundational tests for Phase5-A HSIR formulas."""
from __future__ import annotations

import argparse
from pathlib import Path

from audit_phase5_hsir import OUTPUT_ROOT, run_unit_tests, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT / "UNIT_TESTS.json")
    args = parser.parse_args()
    result = run_unit_tests()
    write_json(args.output, result)
    print(f"STATUS: unit tests {result['status']}")
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
