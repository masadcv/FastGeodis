from gettext import install
import glob
import os
import re
import sys
import warnings

import pkg_resources
from setuptools import find_packages, setup

FORCE_CUDA = os.getenv("FORCE_CUDA", "0") == "1"

BUILD_CPP = BUILD_CUDA = False
TORCH_VERSION = 0
try:
    import torch

    print(f"setup.py with torch {torch.__version__}")
    from torch.utils.cpp_extension import BuildExtension, CppExtension

    BUILD_CPP = True
    from torch.utils.cpp_extension import CUDA_HOME, CUDAExtension

    BUILD_CUDA = (CUDA_HOME is not None) if torch.cuda.is_available() else FORCE_CUDA

    _pt_version = pkg_resources.parse_version(torch.__version__)._version.release
    if _pt_version is None or len(_pt_version) < 3:
        raise AssertionError("unknown torch version")
    TORCH_VERSION = (
        int(_pt_version[0]) * 10000 + int(_pt_version[1]) * 100 + int(_pt_version[2])
    )
except (ImportError, TypeError, AssertionError, AttributeError) as e:
    warnings.warn(f"extension build skipped: {e}")
finally:
    print(
        f"BUILD_CPP={BUILD_CPP}, BUILD_CUDA={BUILD_CUDA}, TORCH_VERSION={TORCH_VERSION}."
    )


def torch_parallel_backend():
    try:
        match = re.search(
            "^ATen parallel backend: (?P<backend>.*)$",
            torch._C._parallel_info(),
            re.MULTILINE,
        )
        if match is None:
            return None
        backend = match.group("backend")
        if backend == "OpenMP":
            return "AT_PARALLEL_OPENMP"
        if backend == "native thread pool":
            return "AT_PARALLEL_NATIVE"
        if backend == "native thread pool and TBB":
            return "AT_PARALLEL_NATIVE_TBB"
    except (NameError, AttributeError):  # no torch or no binaries
        warnings.warn("Could not determine torch parallel_info.")
    return None


def omp_flags():
    if sys.platform == "win32":
        return ["/openmp"]
    if sys.platform == "darwin":
        # https://stackoverflow.com/questions/37362414/
        # return ["-fopenmp=libiomp5"]
        return []
    return ["-fopenmp"]


def get_extensions():
    # this_dir = os.path.dirname(os.path.abspath(__file__))
    # ext_dir = os.path.join(this_dir, "src")
    ext_dir = "FastGeodis"
    include_dirs = [ext_dir]

    source_cpu = glob.glob(os.path.join(ext_dir, "**", "*.cpp"), recursive=True)
    source_cuda = glob.glob(os.path.join(ext_dir, "**", "*.cu"), recursive=True)

    extension = None
    define_macros = [(f"{torch_parallel_backend()}", 1)]
    extra_compile_args = {}
    extra_link_args = []
    sources = source_cpu
    if BUILD_CPP:
        extension = CppExtension
        extra_compile_args.setdefault("cxx", [])
        if torch_parallel_backend() == "AT_PARALLEL_OPENMP":
            extra_compile_args["cxx"] += omp_flags()
        extra_link_args = omp_flags()
    if BUILD_CUDA:
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args = {"cxx": [], "nvcc": []}
        if torch_parallel_backend() == "AT_PARALLEL_OPENMP":
            extra_compile_args["cxx"] += omp_flags()
    if extension is None or not sources:
        return []  # compile nothing
    
    # compile release
    extra_compile_args["cxx"] += ["-g0"]
    
    ext_modules = [
        extension(
            name="FastGeodisCpp",
            sources=sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ]
    return ext_modules


with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fp:
    install_requires = fp.read().splitlines()

# add dependencies folder in include path
dep_dir = os.path.join(".", "dependency")

setup(
    name="FastGeodis",
    version="1.0.4",
    description="Fast Implementation of Generalised Geodesic Distance Transform for CPU (OpenMP) and GPU (CUDA)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/masadcv/FastGeodis",
    author="Muhammad Asad",
    author_email="masadcv@gmail.com",
    license="BSD-3-Clause License",
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
    ],
    install_requires=install_requires,
    cmdclass={
        "build_ext": BuildExtension
    },  # .with_options(no_python_abi_suffix=True)},
    packages=find_packages(exclude=("data", "docs", "examples", "scripts", "tests")),
    zip_safe=False,
    ext_modules=get_extensions(),
    include_dirs=[dep_dir],
)
