try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(name='tcc',
      version='0.1',
      author="Felipe",
      description="""python code""",
      ext_package=["brain"],
      packages=["brain"],
      )
