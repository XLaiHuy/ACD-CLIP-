"""Collision-safe runner for the frozen Trust-v2 M4 recovery v2 audit."""
from __future__ import annotations

import json

from sabra.trust_v2 import visa_audit


def run() -> dict[str, object]:
    visa_audit.configure_output_root(visa_audit.RECOVERY_ROOT)
    return visa_audit.run()


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True))
