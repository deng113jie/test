from setuptools import setup
from Cython.Build import cythonize
import os

setup(
    if os.name == 'nt':
        srcpath= 'src\\helloworld.pyx'
    else:
        srcpath= 'src/helloworld.pyx'
    
    name="cytest",
    version="0.0.1",
    ext_modules = cythonize(srcpath)
)