'''
python setup.py build_ext -i
to compile
'''

from setuptools import dist

dist.Distribution().fetch_build_eggs(['Cython>=0.29.32', 'numpy>=1.21.0'])

from distutils.core import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy


setup(
    name='sim3dr_cython',  # not the package name
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("sim3dr_cython",
                           sources=["lib/rasterize.pyx", "lib/rasterize_kernel.cpp"],
                           language='c++',
                           include_dirs=[numpy.get_include()],
                           extra_compile_args=["-std=c++11"])],
)
