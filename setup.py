from setuptools import setup
from Cython.Build import cythonize

setup(
    name="cytest",
    version="0.0.1",
    ext_modules = cythonize("C:\\Users\\jie\\Documents\\ExeTera-master\\cytest\\src\\helloworld.pyx")
)