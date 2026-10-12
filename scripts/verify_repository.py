"""Read-only Git-index audit. Reports findings without printing suspected secrets."""

import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_REMOTE = "https://github.com/zaidbinnaveed/AI-Engineering-Assignment-01.git"
EVIDENCE_PNGS = {
    "report/evidence/F01_manual_attention.png",
    "report/evidence/F02_causal_mask.png",
    "report/evidence/F04_positional_encoding.png",
    "report/evidence/F05_model_summary.png",
    "report/evidence/F06_loss.png",
    "report/evidence/F07_trained_attention.png",
    "report/evidence/F08_samples.png",
    "report/evidence/F09_ablations.png",
    "report/evidence/F10_inference.png",
}
EVIDENCE_TEXT = {"report/evidence/F03_multihead_shape_trace.txt"}


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT)


def main() -> int:
    findings: list[dict[str, str]] = []
    for line in git("remote", "-v").decode().splitlines():
        name, url, _ = line.split()
        if name != "origin" or url.removesuffix(".git") != EXPECTED_REMOTE.removesuffix(".git"):
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
        if name in EVIDENCE_PNGS or name in EVIDENCE_TEXT:
            if (
                len(content) > 500_000
                or (name in EVIDENCE_PNGS and not content.startswith(b"\x89PNG\r\n\x1a\n"))
                or (name in EVIDENCE_TEXT and b"\0" in content)
            ):
                findings.append({"file": name, "reason": "invalid or oversized evidence artifact"})
            receipt_path = str(Path(name).with_suffix(".json")).replace("\\", "/")
            if receipt_path not in paths:
                findings.append({"file": name, "reason": "missing evidence provenance receipt"})
            else:
                receipt = json.loads(git("show", f":{receipt_path}"))
                if (
                    receipt.get("sha256") != hashlib.sha256(content).hexdigest()
                    or receipt.get("bytes") != len(content)
                    or receipt.get("artifact") != name
                ):
                    findings.append({"file": name, "reason": "evidence content hash mismatch"})
                sources = receipt.get("source_sha256", {})
                if not sources:
                    findings.append({"file": name, "reason": "missing evidence source hashes"})
                for source, digest in sources.items():
                    if (
                        source not in paths
                        or hashlib.sha256(git("show", f":{source}")).hexdigest() != digest
                    ):
                        findings.append({"file": name, "reason": "evidence source hash mismatch"})
        elif len(content) > 1_000_000 or b"\0" in content:
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
