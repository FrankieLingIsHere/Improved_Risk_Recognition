"""
Setup script for the Risk Recognition System.
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="risk-recognition",
    version="0.1.0",
    author="Frankie Ling",
    author_email="",
    description="A BERT-based system for accident risk classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FrankieLingIsHere/Improved_Risk_Recognition",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "flake8>=3.8",
            "black>=21.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "risk-recognition=main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)