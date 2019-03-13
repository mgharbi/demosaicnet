import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

print(setuptools.find_packages())

setuptools.setup(
    name='demosaicnet',
    version='1.0',
    scripts=['demo'] ,
    author="Michaël Gharbi",
    author_email="gharbi@csail.mit.edu",
    description="Minimal implementation of Deep Joint Demosaicking and Denoising [Gharbi2016]",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mgharbi/",
    packages=["demosaicnet"],
    packages_data={"demosaicnet": "data/*"},
    # packages=setuptools.find_packages(),
    classifiers=[
      "Programming Language :: Python :: 3",
      "License :: OSI Approved :: MIT License",
      "Operating System :: MacOS :: MacOS X",
      "Operating System :: POSIX",
    ],
)
