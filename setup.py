from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy
import os
import sys
import mpi4py

def check_for_lib(lib_name):
    import ctypes.util
    if ctypes.util.find_library(lib_name) is None:
        print(f"Error: {lib_name} not found. Please run: conda install -c conda-forge {lib_name}")
        sys.exit(1)

# Only check when actually building the extension
if "build_ext" in sys.argv:
    check_for_lib("pexsi")
    check_for_lib("superlu_dist")
    check_for_lib("parmetis")
    
# 1. Detect the Conda or System prefix to find headers/libraries
conda_prefix = os.environ.get('CONDA_PREFIX')
if not conda_prefix:
    # Fallback to standard paths if not in a conda env
    include_dirs = [mpi4py.get_include(), numpy.get_include()]
    library_dirs = []
else:
    include_dirs = [
        mpi4py.get_include(),
        numpy.get_include(),
        os.path.join(conda_prefix, 'include'),
        os.path.join(conda_prefix, 'include', 'pexsi')
    ]
    library_dirs = [os.path.join(conda_prefix, 'lib')]

ext_modules = [
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

setup(
    name="tb_solve",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=cythonize(ext_modules, language_level="3"),
    install_requires=[
        "numpy",
        "scipy",
        "torch",
    ],
)
