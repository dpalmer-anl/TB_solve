from setuptools import setup, Extension, find_packages
import os
import sys

try:
    from Cython.Build import cythonize
except ImportError:
    cythonize = None


def check_for_lib(lib_name):
    import ctypes.util

    if ctypes.util.find_library(lib_name) is None:
        print(f"Error: {lib_name} not found. Please run: conda install -c conda-forge {lib_name}")
        sys.exit(1)


def build_extensions():
    # Build the native extension only when explicitly requested.
    if os.environ.get("TB_SOLVE_BUILD_EXT") != "1":
        return []

    try:
        import numpy
    except ImportError as exc:
        raise RuntimeError("numpy is required to build extensions") from exc

    try:
        import mpi4py
    except ImportError as exc:
        raise RuntimeError("mpi4py is required to build extensions") from exc

    # Only check when actually building the extension
    if "build_ext" in sys.argv:
        check_for_lib("pexsi")
        check_for_lib("superlu_dist")
        check_for_lib("parmetis")

    # 1. Detect the Conda or System prefix to find headers/libraries
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        # Fallback to standard paths if not in a conda env
        include_dirs = [mpi4py.get_include(), numpy.get_include()]
        library_dirs = []
    else:
        include_dirs = [
            mpi4py.get_include(),
            numpy.get_include(),
            os.path.join(conda_prefix, "include"),
            os.path.join(conda_prefix, "include", "pexsi"),
        ]
        library_dirs = [os.path.join(conda_prefix, "lib")]

    return [
        Extension(
            "tb_solve.cython_scripts.pexsi_wrapper",
            sources=["src/tb_solve/cython_scripts/pexsi_wrapper.pyx"],
            include_dirs=include_dirs,
            libraries=["pexsi", "superlu_dist", "parmetis"],
            library_dirs=library_dirs,
            extra_compile_args=["-std=c++11"],
            language="c++",
        ),
    ]


ext_modules = build_extensions()

setup(
    name="tb_solve",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=cythonize(ext_modules, language_level="3") if ext_modules else [],
    install_requires=[
        "numpy",
        "scipy",
        "torch",
    ],
)
