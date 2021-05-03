from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy

setup(name="clsh",
      ext_modules=cythonize(Extension(
          "clsh",  # the extension name
          sources=["clsh.pyx", "LSH.cpp"],  # the Cython source and additional C++ source files
          language="c++",  # generate and compile C++ code
          include_dirs=[numpy.get_include()],
          extra_compile_args=["-std=c++11"]
      )))
