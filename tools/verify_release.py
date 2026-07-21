#!/usr/bin/env python3
"""Fail closed on model identity and release-boundary violations."""

from __future__ import annotations

import hashlib
from pathlib import Path

EXPECTED_MODEL = "823a305b9cdf6d16e2296644d88a343dfc68621fa8e3d4b3c0125772c50b92e8"
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".ckpt", ".npz"}
FORBIDDEN_NAMES = {"legacy.py", "benchmark_legacy_codecs.py"}


def main():
    root = Path(__file__).resolve().parents[1]
    weights = root / "xjpeg" / "weights" / "model.safetensors"
    actual = hashlib.sha256(weights.read_bytes()).hexdigest()
    if actual != EXPECTED_MODEL:
        raise SystemExit(f"model SHA-256 mismatch: {actual}")
    violations = [
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_file()
        and (path.suffix.lower() in FORBIDDEN_SUFFIXES or path.name in FORBIDDEN_NAMES)
        and not any(part in {".git", ".venv", "build", "dist"} for part in path.parts)
    ]
    if violations:
        raise SystemExit(f"private/unsafe artifacts in release tree: {violations}")
    print(f"release boundary OK; model sha256={actual}")


if __name__ == "__main__":
    main()

