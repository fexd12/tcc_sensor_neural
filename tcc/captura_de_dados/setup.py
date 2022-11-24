try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(name='capturaDeDados',
      version='0.1',
      author="Felipe",
      description="""python code""",
      ext_package=["brain"],
      packages=["brain"],
      )
