#!/usr/bin/env python3
"""Build the pinned static MozJPEG cjpeg executable for the current host."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

COMMIT = "08265790774cd0714832c9e675522acbe5581437"
REPOSITORY = "https://github.com/mozilla/mozjpeg.git"


def run(*command, cwd=None):
    subprocess.run(command, cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("xjpeg/bin") / ("cjpeg.exe" if os.name == "nt" else "cjpeg"),
    )
    parser.add_argument("--commit", default=COMMIT)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="xjpeg-mozjpeg-build-") as root:
        root = Path(root)
        source = root / "source"
        build = root / "build"
        run("git", "clone", "--filter=blob:none", REPOSITORY, str(source))
        run("git", "checkout", args.commit, cwd=source)
        configure = [
            "cmake", "-S", str(source), "-B", str(build),
            "-DBUILD_SHARED_LIBS=OFF",
            "-DWITH_JPEG8=ON",
            "-DWITH_TURBOJPEG=OFF",
            "-DWITH_SIMD=ON",
            "-DPNG_SUPPORTED=OFF",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded",
        ]
        if sys.platform == "darwin":
            deployment_target = os.environ.get(
                "MACOSX_DEPLOYMENT_TARGET", "12.0"
            )
            configure.append(
                f"-DCMAKE_OSX_DEPLOYMENT_TARGET={deployment_target}"
            )
        run(*configure)
        run(
            "cmake", "--build", str(build), "--config", "Release",
            "--target", "cjpeg-static", "--parallel", "2",
        )
        names = ("cjpeg-static.exe", "cjpeg.exe") if os.name == "nt" else ("cjpeg-static",)
        matches = [
            path for name in names for path in build.rglob(name)
            if path.is_file()
        ]
        if not matches:
            raise SystemExit("built cjpeg executable was not found")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(matches[0], args.output)
        if os.name != "nt":
            args.output.chmod(0o755)
        run(str(args.output), "-version")
        print(args.output)


if __name__ == "__main__":
    main()
