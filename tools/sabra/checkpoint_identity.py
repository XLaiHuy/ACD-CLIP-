"""Print and validate the identity of one explicit training checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from model.checkpoint_loader import checkpoint_identity  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--expected-epoch", type=int, default=None)
args = parser.parse_args()
print(json.dumps(checkpoint_identity(args.checkpoint, expected_epoch=args.expected_epoch), indent=2, sort_keys=True))
