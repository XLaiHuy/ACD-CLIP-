#!/usr/bin/env python3
"""Stage 1.6 K1 plus exact-base NO-OP zero-step wrapper."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audit_p4_semantic_interface import main


main(default_k1=True, default_noop=True)
