#!/usr/bin/env python
# -*- coding: utf-8 -*-

import setuptools
import sys
import glob
import os
from setuptools import Extension
from setuptools.command.build_ext import build_ext as _build_ext
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension


class CustomBuildExt(_build_ext):
    """CustomBuildExt"""

    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy

        self.include_dirs.append(numpy.get_include())


class BuildExt(BuildExtension):
    """CustomBuildExt"""

    def finalize_options(self):
        super().finalize_options()
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy

        self.include_dirs.append(numpy.get_include())


compile_extra_args = ["-std=c++11", "-O3", "-fopenmp"]

link_extra_args = ["-fopenmp"]
if sys.platform.startswith("darwin"):
    compile_extra_args = ["-std=c++11", "-mmacosx-version-min=10.9"]
    link_extra_args = ["-stdlib=libc++", "-mmacosx-version-min=10.9"]

############################
# Check Platform
############################

assert sys.platform.startswith("linux") or sys.platform.startswith(
    "darwin"
), "Only Supported On Linux And Darwin"

# setup(
#     name="xgnn",
#     version="1.0",
#     description="A distributed GNN system",
#     author="xin ning",
#     author_email="ningxin1009@163.com",
#     packages=["xgnn"],
# )

def find_cuda():
    # TODO: find cuda
    home = os.getenv("CUDA_HOME")
    path = os.getenv("CUDA_PATH")
    if home is not None:
        return home
    elif path is not None:
        return path
    else:
        return '/usr/local/cuda'


def have_cuda():
    import torch
    return torch.cuda.is_available()


def create_extension(with_cuda=False):
    print('Building torch_quiver with CUDA:', with_cuda)
    srcs = []
    srcs += glob.glob('ccsrc/propeller/*.cu')
    srcs += glob.glob('ccsrc/propeller/*.cpp')

    include_dirs = [
        os.path.join(os.getcwd(), 'ccsrc/propeller/include')
    ]
    # print(include_dirs)
    
    library_dirs = []
    libraries = []
    extra_cxx_flags = [
        '-std=c++17',
        # TODO: enforce strict build
        # '-Wall',
        # '-Werror',
        # '-Wfatal-errors',
    ]
    if with_cuda:
        cuda_home = find_cuda()
        include_dirs += [os.path.join(cuda_home, 'include')]
        library_dirs += [os.path.join(cuda_home, 'lib64')]
        extra_cxx_flags += ['-DHAVE_CUDA']

    if os.getenv('QUIVER_ENABLE_TRACE'):
        extra_cxx_flags += ['-DQUIVER_ENABLE_TRACE=1']

    return CppExtension(
        'propeller',
        srcs,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        # with_cuda=with_cuda,
        extra_compile_args={
            'cxx': extra_cxx_flags,
            'nvcc': ['-O3', '--expt-extended-lambda', '-lnuma'],
        },
    )

setuptools.setup(
    name="xgnn",
    version="0.1",
    author="ning xin",
    description="A distributed GNN system",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=["scikit-learn>=0.24.2", "Cython"],
    zip_safe=False,
    cmdclass={"build_ext": BuildExt},  # "build_ext": CustomBuildExt,
    include_package_data=True,
    ext_modules=[
        Extension(
            "sample_kernel",
            ["xgnn/cpython/sample_kernel.pyx",],
            language="c++",
            extra_compile_args=compile_extra_args,
            extra_link_args=link_extra_args,
        ),
        CUDAExtension(
            name="xgnn_kernel",
            sources=["ccsrc/kernel/xgnn.cpp", "ccsrc/kernel/xgnn_kernel.cu"],
        ),
        CppExtension(
            name="rabbit",
            sources=["ccsrc/reorder/reorder.cpp"],
            extra_compile_args=["-O3", "-fopenmp", "-mcx16"],
            libraries=["numa", "tcmalloc_minimal"],
        ),
        create_extension(have_cuda())
    ],
    ext_package="xgnn",
)
