from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    readme = fh.read()

version = {}
with open("butterpy/version.py") as fp:
    exec(fp.read(), version)

setup(
    name="butterpy",
    version=version["__version__"],
    author="Zachary R. Claytor <zclaytor@hawaii.edu> and Miles Lucas <mdlucas@hawaii.edu>",
    description="Tools for simulating stellar rotational light curves using realistic spot evolution",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/zclaytor/butterpy",
    license="MIT",
    python_requires="==3.*,>=3.6.0",
    packages=find_packages(),
    install_requires=["astropy", "numpy", "scipy", "matplotlib", "tqdm", "pandas"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Unix",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
)