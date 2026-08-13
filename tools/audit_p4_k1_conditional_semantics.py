#!/usr/bin/env python3
"""K1 zero-step audit using the Stage-0 interface checks and K1 terminal."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_p4_semantic_interface import main

if __name__ == "__main__":
    main(default_k1=True)
