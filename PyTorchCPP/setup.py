from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='geodis',
      ext_modules=[cpp_extension.CppExtension('geodis', ['geodis.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})