import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='demosaicnet',
    version='1.0',
    scripts=["scripts/demosaicnet_demo"],
    author="Michaël Gharbi",
    author_email="gharbi@csail.mit.edu",
    description="Minimal implementation of Deep Joint Demosaicking and Denoising [Gharbi2016]",
    long_description=long_description,
    url="https://github.com/mgharbi/",
    packages=["demosaicnet"],
    include_package_data=True,
    classifiers=[
      "Programming Language :: Python :: 3",
      "License :: OSI Approved :: MIT License",
      "Operating System :: MacOS :: MacOS X",
      "Operating System :: POSIX",
    ],
)
