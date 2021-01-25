import re
import setuptools


with open('demosaicnet/version.py') as fid:
    try:
        __version__, = re.findall( '__version__ = "(.*)"', fid.read() )
    except:
        raise ValueError("could not find version number")


with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name='demosaicnet',
    version=__version__,
    scripts=["scripts/demosaicnet_demo"],
    author="Michaël Gharbi",
    author_email="gharbi@csail.mit.edu",
    description="Minimal implementation of Deep Joint Demosaicking and Denoising [Gharbi2016]",
    long_description=long_description,
    url="https://github.com/mgharbi/",
    packages = setuptools.find_packages(exclude=["tests"]),
    include_package_data=True,
    install_requires=["wget", "torch-tools"],
    classifiers=[
      "Programming Language :: Python :: 3",
      "License :: OSI Approved :: MIT License",
      "Operating System :: MacOS :: MacOS X",
      "Operating System :: POSIX",
    ],
)
