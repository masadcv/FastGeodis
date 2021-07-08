from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='geodis',
      ext_modules=[cpp_extension.CppExtension(
            name='geodis', 
            sources=['geodis.cpp'],
            extra_compile_args=['-fopenmp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})