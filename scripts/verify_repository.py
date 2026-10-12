"""Read-only Git-index audit. Reports findings without printing suspected secrets."""

import ast
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_REMOTE = "https://github.com/zaidbinnaveed/AI-Engineering-Assignment-01.git"


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT)


def main() -> int:
    findings: list[dict[str, str]] = []
    for line in git("remote", "-v").decode().splitlines():
        name, url, _ = line.split()
        if name != "origin" or url != EXPECTED_REMOTE:
            findings.append({"file": "git remote", "reason": "unexpected destination"})
    paths = [name.decode("utf-8") for name in git("ls-files", "-z").split(b"\0") if name]
    if not paths:
        findings.append({"file": "index", "reason": "empty index"})
    patterns = [
        rb"gh[pousr]_[A-Za-z0-9]{30,}",
        rb"github_pat_[A-Za-z0-9_]{30,}",
        rb"AKIA[0-9A-Z]{16}",
        rb"sk-[A-Za-z0-9_-]{24,}",
        rb"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----",
    ]
    max_bytes = 0
    for name in paths:
        content = git("show", f":{name}")
        max_bytes = max(max_bytes, len(content))
        if len(content) > 1_000_000 or b"\0" in content:
            findings.append({"file": name, "reason": "large or binary content"})
        if any(re.search(pattern, content) for pattern in patterns):
            findings.append({"file": name, "reason": "possible credential/private key"})
        relative = Path(name)
        if relative.suffix.lower() in {".pt", ".pth", ".ckpt", ".safetensors", ".pem", ".key"}:
            findings.append({"file": name, "reason": "forbidden artifact extension"})
        if ".venv" in relative.parts or (
            relative.name.startswith(".env") and name != ".env.example"
        ):
            findings.append({"file": name, "reason": "environment/secret file tracked"})
        if relative.suffix == ".py":
            tree = ast.parse(content.decode("utf-8"), filename=name)
            for node in ast.walk(tree):
                imported = []
                if isinstance(node, ast.Import):
                    imported = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    imported = [node.module or ""]
                if any(module.split(".")[0] == "transformers" for module in imported):
                    findings.append({"file": name, "reason": "pretrained-model library import"})
    print(
        json.dumps(
            {
                "status": "PASS" if not findings else "FAIL",
                "tracked_files": len(paths),
                "largest_file_bytes": max_bytes,
                "findings": findings,
                "scope": "Git-index size/binary/credential patterns/imports/remote audit; manual provenance review also required",
            },
            indent=2,
        )
    )
    return bool(findings)


if __name__ == "__main__":
    raise SystemExit(main())
