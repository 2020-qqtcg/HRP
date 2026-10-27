"""Additive asset pin: never replace the v1 manifest or frontend."""

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

from ..assets import VENDOR

MANIFEST = json.loads(Path(__file__).with_name("upstream.json").read_text())


def verify_assets(fetch=False):
    for relative, expected in MANIFEST["files"].items():
        path = VENDOR / relative
        if fetch and not path.exists():
            data = urlopen(f'{MANIFEST["raw_url"]}/{MANIFEST["revision"]}/{relative}', timeout=60).read()
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f"Upstream checksum mismatch: {relative}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Asset checksum mismatch: {relative}")
