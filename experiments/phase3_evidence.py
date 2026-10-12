"""Small text/PNG evidence copies with exact source and artifact hashes."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def save_evidence(path, *, sources, details):
    content = path.read_bytes()
    if len(content) > 500_000:
        raise ValueError("evidence must be no larger than 500 KB")
    if path.suffix == ".png":
        if not content.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("invalid PNG")
    elif path.suffix == ".txt":
        content.decode("utf-8")
    else:
        raise ValueError("only text and PNG evidence are supported")
    destination = ROOT / "report/evidence" / path.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    receipt = {
        "artifact": destination.relative_to(ROOT).as_posix(),
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
        "source_sha256": {
            source: hashlib.sha256((ROOT / source).read_bytes()).hexdigest() for source in sources
        },
        "details": details,
    }
    destination.with_suffix(".json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    return destination
