from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    readme = fh.read()

exec(open("butterpy/version.py").read())

setup(
    description="Tools for simulating stellar rotational light curves using realistic spot evolution",
    long_description=readme,
    long_description_content_type="text/markdown",
    name="butterpy",
    version=__version__,
    python_requires="==3.*,>=3.6.0",
    author="Zachary Claytor <zclaytor@hawaii.edu> and Miles Lucas <mdlucas@hawaii.edu>",
    packages=find_packages(),
    install_requires=["astropy", "numpy", "scipy", "matplotlib", "tqdm", "pandas", "cartopy"],
)
