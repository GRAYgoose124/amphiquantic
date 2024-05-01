from setuptools import setup, find_packages

setup(
    name="amphiquantic",
    version="0.1.0",
    packages=find_packages(),
    description="A molecular visualization tool for PDB files.",
    author="Your Name",
    author_email="your.email@example.com",
    install_requires=["numpy", "matplotlib", "mpl_toolkits"],
)
