"""
Setup script for pyszo - Python bindings for SZo
Automatically downloads and builds SZo with bundled zstd.
"""

import sys
import subprocess
import shutil
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext
from Cython.Build import cythonize
import numpy as np



SZo_VERSION = "3.3.1"

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
        szo_dir = build_temp / "SZo"
        
        if (szo_dir / "build" / "include" / "SZo" / "version.hpp").exists():
            print(f"SZo already built at: {szo_dir}")
            return szo_dir
        
        if not szo_dir.exists():
            print(f"Cloning SZo v{SZo_VERSION}...")
            subprocess.run([
                "git", "clone", "--depth", "1",
                "--branch", f"v{SZo_VERSION}",
                "--single-branch",
                "https://github.com/szcompressor/SZo.git",
                str(szo_dir)
            ], check=True)

        build_dir = szo_dir / "build"
        build_dir.mkdir(exist_ok=True)
        
        cmake_args = ["cmake"]
        cmake_args.extend([
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_TESTING=OFF",
            "-DBUILD_SZo_BINARY=OFF",
            "-DSZo_USE_BUNDLED_ZSTD=ON",
            ".."
        ])
        subprocess.run(cmake_args, cwd=build_dir, check=True)
        subprocess.run(["cmake", "--build", ".", "-j"], cwd=build_dir, check=True)
        print(f"Built SZo v{SZo_VERSION}")
        return szo_dir




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
            "pyszo.sz",
            sources=["src/pyszo/sz.pyx"],
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
