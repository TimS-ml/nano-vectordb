"""
Setup configuration for the nano-vectordb package.

This script handles the package installation, including:
- Reading package metadata from __init__.py
- Loading dependencies from requirements.txt
- Configuring PyPI package information
"""

import setuptools

# Read the README file for the long description on PyPI
with open("./readme.md", "r") as fh:
    long_description = fh.read()

# Extract version, author, and URL from __init__.py to maintain single source of truth
vars2find = ["__author__", "__version__", "__url__"]
vars2readme = {}
with open("./nano_vectordb/__init__.py") as f:
    for line in f.readlines():
        for v in vars2find:
            if line.startswith(v):
                # Parse the variable assignment (e.g., __version__ = "0.0.4.3")
                line = line.replace(" ", "").replace('"', "").replace("'", "").strip()
                vars2readme[v] = line.split("=")[1]

# Read package dependencies from requirements.txt
deps = []
with open("./requirements.txt") as f:
    for line in f.readlines():
        if not line.strip():  # Skip empty lines
            continue
        deps.append(line.strip())

# Configure the package for distribution
setuptools.setup(
    name="nano-vectordb",
    url=vars2readme["__url__"],
    version=vars2readme["__version__"],
    author=vars2readme["__author__"],
    description="A simple, easy-to-hack Vector Database implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["nano_vectordb"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",  # Minimum Python version required
    install_requires=deps,  # Package dependencies
)
