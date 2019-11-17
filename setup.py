from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    readme = fh.read()

setup(
    long_description=readme,
    long_description_content_type="text/markdown",
    name="butterpy",
    version="0.1.0",
    python_requires="==3.*,>=3.6.0",
    author="Zachary Claytor <zclaytor@hawaii.edu> and Miles Lucas <mdlucas@hawaii.edu>",
    packages=find_packages(),
    install_requires=["astropy", "numpy", "scipy", "matplotlib", "tqdm", "pandas"],
)
