from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='FastGeodis',
      ext_modules=[cpp_extension.CppExtension(
            name='FastGeodis', 
            sources=['fastgeodis.cpp'],
            extra_compile_args=['-fopenmp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})