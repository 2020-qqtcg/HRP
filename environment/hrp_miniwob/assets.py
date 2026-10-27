"""Fetch and verify the pinned, unmodified MiniWoB++ frontend assets."""

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "miniwob"
MANIFEST = json.loads(Path(__file__).with_name("upstream.json").read_text())


def fetch_assets() -> None:
    for relative, expected in MANIFEST["files"].items():
        path = VENDOR / relative
        if not path.exists():
            url = f'{MANIFEST["raw_url"]}/{MANIFEST["revision"]}/{relative}'
            data = urlopen(url, timeout=60).read()
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f"Upstream checksum mismatch: {relative}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    verify_assets()


def verify_assets() -> None:
    for relative, expected in MANIFEST["files"].items():
        path = VENDOR / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Local asset differs from pinned upstream: {relative}")
