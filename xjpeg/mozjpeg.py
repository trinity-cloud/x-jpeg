"""MozJPEG discovery and custom-quantization-table encoding.

Published X-JPEG platform wheels carry a statically linked ``cjpeg`` binary.
Source installs may instead provide one through ``XJPEG_MOZJPEG`` or PATH.
The emitted file is an ordinary JPEG; MozJPEG is needed only while encoding.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image


class MozJPEGNotFoundError(RuntimeError):
    """Raised when the MozJPEG backend is explicitly requested but missing."""


def _binary_name() -> str:
    return "cjpeg.exe" if os.name == "nt" else "cjpeg"


def _is_mozjpeg(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        result = subprocess.run(
            [str(path), "-version"], capture_output=True, timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    version = (result.stdout + result.stderr).decode(
        "utf-8", errors="replace"
    ).lower()
    return result.returncode == 0 and "mozjpeg" in version


def _materialize_packaged_binary(path: Path) -> Path:
    """Copy a non-executable wheel payload to a verified user cache."""
    if os.name == "nt" or os.access(path, os.X_OK):
        return path
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    cache_root = Path(
        os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
    )
    cached = cache_root / "xjpeg" / "mozjpeg" / digest / _binary_name()
    if not cached.exists():
        cached.parent.mkdir(parents=True, exist_ok=True)
        temporary = cached.with_suffix(".tmp")
        shutil.copyfile(path, temporary)
        temporary.chmod(0o755)
        temporary.replace(cached)
    return cached


@lru_cache(maxsize=8)
def find_mozjpeg(explicit: str | os.PathLike | None = None) -> Path | None:
    """Return a verified MozJPEG ``cjpeg`` executable, if available."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    environment = os.environ.get("XJPEG_MOZJPEG")
    if environment:
        candidates.append(Path(environment).expanduser())

    packaged = Path(__file__).parent / "bin" / _binary_name()
    if packaged.is_file():
        candidates.append(_materialize_packaged_binary(packaged))

    for name in ("mozjpeg", "cjpeg"):
        resolved = shutil.which(name)
        if resolved:
            candidates.append(Path(resolved))

    if os.name != "nt":
        candidates.extend([
            Path("/opt/homebrew/opt/mozjpeg/bin/cjpeg"),
            Path("/usr/local/opt/mozjpeg/bin/cjpeg"),
        ])

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate not in seen and _is_mozjpeg(candidate):
            return candidate
        seen.add(candidate)
    return None


def mozjpeg_version(path: str | os.PathLike | None = None) -> str | None:
    binary = find_mozjpeg(path)
    if binary is None:
        return None
    result = subprocess.run(
        [str(binary), "-version"], capture_output=True, check=True
    )
    return (result.stdout + result.stderr).decode("utf-8").strip()


def _write_qtables(path: Path, tables: tuple[np.ndarray, ...]) -> None:
    path.write_text(
        "\n\n".join(
            "\n".join(
                " ".join(str(int(value)) for value in row)
                for row in table
            )
            for table in tables
        )
        + "\n",
        encoding="ascii",
    )


def encode_mozjpeg(
    image: Image.Image,
    luma_table,
    cb_table,
    cr_table=None,
    *,
    subsampling: int = 2,
    optimize: bool = True,
    progressive: bool = True,
    executable: str | os.PathLike | None = None,
) -> bytes:
    """Encode RGB pixels with learned DQTs and MozJPEG trellis optimization."""
    binary = find_mozjpeg(executable)
    if binary is None:
        raise MozJPEGNotFoundError(
            "MozJPEG was not found. Install a platform wheel, set "
            "XJPEG_MOZJPEG=/path/to/cjpeg, or use backend='pillow'."
        )
    sampling = {
        0: "1x1,1x1,1x1",  # 4:4:4
        1: "2x1,1x1,1x1",  # 4:2:2
        2: "2x2,1x1,1x1",  # 4:2:0
    }
    if subsampling not in sampling:
        raise ValueError("subsampling must be 0 (444), 1 (422), or 2 (420)")

    tables = [luma_table, cb_table]
    if cr_table is not None:
        tables.append(cr_table)
    rounded = tuple(
        np.clip(np.rint(np.asarray(table)), 1, 255).astype(np.int64)
        for table in tables
    )

    with tempfile.TemporaryDirectory(prefix="xjpeg-mozjpeg-") as directory:
        directory = Path(directory)
        source = directory / "source.ppm"
        qtables = directory / "tables.txt"
        output = directory / "output.jpg"
        image.convert("RGB").save(source, format="PPM")
        _write_qtables(qtables, rounded)

        command = [
            str(binary),
            "-dct", "float",
            "-tune-ms-ssim",
            "-sample", sampling[subsampling],
            "-qslots", "0,1,2" if len(rounded) == 3 else "0,1,1",
            "-qtables", str(qtables),
            "-outfile", str(output),
        ]
        if optimize:
            command.append("-optimize")
        command.append("-progressive" if progressive else "-baseline")
        command.append(str(source))
        result = subprocess.run(command, capture_output=True, check=False)
        if result.returncode:
            stderr = result.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(
                f"MozJPEG failed with exit code {result.returncode}: {stderr}"
            )
        return output.read_bytes()
