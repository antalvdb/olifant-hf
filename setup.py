"""
Setup script for Olifant HuggingFace Integration
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="olifant-hf",
    version="0.1.0",
    author="Antal van den Bosch",
    description="HuggingFace integration for Olifant (TiMBL-based) language models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/antalvdb/olifant",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
        ],
        "timbl": [
            "python3-timbl>=2023.1.0",
        ],
    },
)
