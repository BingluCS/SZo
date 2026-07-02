"""
Setup script for pyszo - Python bindings for SZo
Automatically downloads and builds SZo with bundled zstd.
"""

import sys
import os
import platform
import subprocess
import shutil
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext
from Cython.Build import cythonize
import numpy as np




class BuildSZoExtension(_build_ext):

    def run(self):
        szo_dir = self.download_and_build_szo()

        for ext in self.extensions:
            ext.include_dirs.insert(0, str(szo_dir / "include"))
            ext.include_dirs.insert(0, str(szo_dir / "build" / "include"))
            ext.include_dirs.append(str(szo_dir / "build" / "_deps" / "zstdfetched-src" / "lib"))
            ext.library_dirs.append(str(szo_dir / "build" / "tools" / "zstd"))
            ext.library_dirs.append(str(szo_dir / "build" / "tools" / "zstd" / "Release"))
            ext.library_dirs.append(str(szo_dir / "build" / "tools" / "zstd" / "Debug"))

        super().run()

        if sys.platform == "darwin":
            zstd_lib_name = "libzstd.dylib"
        elif sys.platform == "win32":
            zstd_lib_name = "zstd.dll"
        else:
            zstd_lib_name = "libzstd.so"

        zstd_base = szo_dir / "build" / "tools" / "zstd"
        package_dir = Path(self.build_lib) / "pyszo"
        if package_dir.exists():
            for subdir in ["", "Release", "Debug"]:
                zstd_lib = zstd_base / subdir / zstd_lib_name
                if zstd_lib.exists():
                    shutil.copy2(zstd_lib, package_dir / zstd_lib.name)
                    print(f"Copied {zstd_lib.name} to package")
                    break

    def download_and_build_szo(self):
        build_temp = Path(self.build_temp).absolute()
        build_temp.mkdir(parents=True, exist_ok=True)

        # Prefer the local SZo checkout when installing from source. This file
        # lives at <SZo>/tools/pyszo/setup.py, so the repo root is two levels up.
        # Using the local tree needs no network and picks up any local changes.
        local_root = Path(__file__).resolve().parent.parent.parent
        if (local_root / "CMakeLists.txt").exists() and (local_root / "include" / "SZo").is_dir():
            szo_dir = local_root
            print(f"Using local SZo source: {szo_dir}")
        else:
            # Fallback (e.g. an sdist with no source tree): fetch main from GitHub.
            szo_dir = build_temp / "SZo"
            if not (szo_dir / "include" / "SZo").is_dir():
                if szo_dir.exists():
                    shutil.rmtree(szo_dir)
                print("Cloning SZo (main)...")
                subprocess.run([
                    "git", "clone", "--depth", "1",
                    "https://github.com/BingluCS/SZo.git",
                    str(szo_dir),
                ], check=True)

        build_dir = szo_dir / "build"
        if (build_dir / "include" / "SZo" / "version.hpp").exists():
            print(f"SZo already built at: {build_dir}")
            return szo_dir

        build_dir.mkdir(exist_ok=True)
        subprocess.run([
            "cmake",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_TESTING=OFF",
            "-DBUILD_SZo_BINARY=OFF",
            "-DSZo_USE_BUNDLED_ZSTD=ON",
            "..",
        ], cwd=build_dir, check=True)
        # pyszo only needs the header-only SZo headers, the generated version.hpp
        # (created at configure time) and the bundled zstd shared library.
        subprocess.run(["cmake", "--build", ".", "--target", "zstd", "-j"], cwd=build_dir, check=True)
        print(f"Built SZo deps at: {build_dir}")
        return szo_dir




def _simd_flags():
    """SIMD compile flags for the header-only SZo kernels.

    SZo's AVX2 / SVE2 code is guarded by ``__AVX2__`` / ``__ARM_FEATURE_SVE2``,
    which the compiler only defines when the right ``-march`` / ``-m`` flag is
    passed to the translation unit that includes the SZo headers (i.e. these
    Cython extensions). Without a flag, SZo falls back to the scalar path.

    Default ('auto') is ``-march=native``: it enables AVX2 on an AVX2-capable
    x86 and SVE2 on an SVE2-capable ARM, and stays scalar on anything else —
    ideal when you build and run on the same machine (e.g. ``pip install -e .``).

    Override with the ``PYSZO_SIMD`` environment variable:

    ==============  ==================================================
    PYSZO_SIMD      effect
    ==============  ==================================================
    auto / native   ``-march=native`` (default) — the build machine's ISA
    avx2            ``-mavx2 -mfma`` — portable AVX2 floor (x86)
    sve2            ``-march=armv8.6-a+sve2`` — ARM SVE2
    none / scalar   no SIMD — portable everywhere, slowest
    <other>         passed verbatim, e.g. PYSZO_SIMD="-march=znver4"
    ==============  ==================================================
    """
    mode = os.environ.get("PYSZO_SIMD", "auto").strip()
    m = mode.lower()

    if sys.platform == "win32":
        # MSVC has no -march=native; AVX2 is the meaningful knob.
        if m in ("none", "scalar"):
            return []
        if m in ("auto", "native", "avx2"):
            return ["/arch:AVX2"] if platform.machine().lower() in ("amd64", "x86_64", "x86") else []
        return []

    if m in ("none", "scalar"):
        return []
    if m in ("auto", "native"):
        return ["-march=native"]
    if m == "avx2":
        return ["-mavx2", "-mfma"]
    if m == "sve2":
        return ["-march=armv8.6-a+sve2"]
    return [mode]  # verbatim override


def create_extensions():
    include_dirs = [np.get_include()]
    library_dirs = []
    libraries = ['zstd']
    extra_compile_args = []
    extra_link_args = []
    
    if sys.platform == 'win32':
        extra_compile_args.extend(['/std:c++17', '/O2'])
    elif sys.platform == 'darwin':
        extra_compile_args.extend(['-std=c++17', '-O3', '-stdlib=libc++', '-mmacosx-version-min=10.9'])
        extra_link_args.extend(['-stdlib=libc++', '-Wl,-rpath,@loader_path'])
    elif sys.platform == 'linux':
        extra_compile_args.extend(['-std=c++17', '-O3'])
        extra_link_args.extend(['-Wl,-rpath,$ORIGIN'])
    
    simd = _simd_flags()
    extra_compile_args.extend(simd)
    print(f"[pyszo] SIMD flags: {' '.join(simd) if simd else '(scalar)'}"
          f"  — override with PYSZO_SIMD (auto|avx2|sve2|none)")

    extensions = [
        Extension(
            "pyszo.pyConfig",
            sources=["src/pyszo/pyConfig.pyx"],
            include_dirs=include_dirs,
            libraries=libraries,
            library_dirs=library_dirs,
            language='c++',
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        ),
        Extension(
            "pyszo.szo",
            sources=["src/pyszo/szo.pyx"],
            include_dirs=include_dirs,
            libraries=libraries,
            library_dirs=library_dirs,
            language='c++',
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        ),
    ]
    
    return cythonize(extensions, compiler_directives={'language_level': '3', 'embedsignature': True})


if __name__ == "__main__":
    setup(
        name="pyszo",
        version="1.0.2",
        packages=["pyszo"],
        package_dir={"": "src"},
        ext_modules=create_extensions(),
        cmdclass={'build_ext': BuildSZoExtension},
        test_suite="tests",
        tests_require=["pytest"],
    )
