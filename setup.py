from setuptools import setup, find_packages

setup(
    name="amphiquantic",
    version="0.1.0",
    packages=find_packages(),
    description="A molecular visualization tool for PDB files.",
    author="Grayson Miller",
    author_email="grayson.miller124@gmail.com",
    install_requires=["numpy", "matplotlib", "mpl_toolkits"],
)
