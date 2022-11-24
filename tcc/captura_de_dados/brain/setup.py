
from distutils.core import setup, Extension

example_module = Extension('_brain',
                           sources=['Brain.cpp', 'Brain.i'],
                           include_dirs=['Brain.h'],
                           language="c++"
                           )

setup(name='brain',
      version='0.1',
      author="SWIG Docs",
      description="""Simple swig example from docs""",
      ext_modules=[example_module],
      py_modules=["brain"]
      )
