from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tools.sabra_v2.p26_parent import require_digest


def test_require_digest_accepts_exact_bytes_and_rejects_substitution(tmp_path: Path) -> None:
    asset = tmp_path / "asset.bin"
    asset.write_bytes(b"p26 frozen asset")
    expected = hashlib.sha256(b"p26 frozen asset").hexdigest()

    assert require_digest(asset, expected) == expected
    with pytest.raises(RuntimeError, match="digest mismatch"):
        require_digest(asset, "0" * 64)
