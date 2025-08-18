from setuptools import setup, find_packages

setup(
    name="amphiquantic",
    version="0.1.0", 
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    description="A python package for molecular modeling.",
    author="Grayson Miller",
    author_email="grayson.miller124@gmail.com",
    install_requires=["numpy", "matplotlib"],
)
