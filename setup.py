# setup.py
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mlip-test",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Testing tools for Machine Learning Interatomic Potentials",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mlip-testing",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "ase>=3.22.0",
        "numpy>=1.20.0",
    ],
    extras_require={
        "mace": ["mace-torch"],
        "dev": ["pytest", "black", "flake8"],
    },
)
