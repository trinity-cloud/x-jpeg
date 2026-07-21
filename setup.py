"""Wheel tagging for optional bundled MozJPEG executables."""

import os
import platform as platform_module
import sys
from pathlib import Path

from setuptools import Distribution, setup
from setuptools.command.bdist_wheel import bdist_wheel


def _has_native_binary() -> bool:
    root = Path(__file__).parent / "xjpeg" / "bin"
    return any((root / name).is_file() for name in ("cjpeg", "cjpeg.exe"))


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return _has_native_binary()


class PlatformWheel(bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        if _has_native_binary():
            self.root_is_pure = False

    def get_tag(self):
        python, abi, wheel_platform = super().get_tag()
        if _has_native_binary():
            if sys.platform == "darwin":
                target = os.environ.get("MACOSX_DEPLOYMENT_TARGET", "12.0")
                target = target.replace(".", "_")
                machine = platform_module.machine().replace("-", "_")
                wheel_platform = f"macosx_{target}_{machine}"
            return "py3", "none", wheel_platform
        return python, abi, wheel_platform


setup(
    distclass=BinaryDistribution,
    cmdclass={"bdist_wheel": PlatformWheel},
)
