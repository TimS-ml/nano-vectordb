"""
Nano Vector Database - A simple, lightweight vector database implementation.

This module provides a minimal yet functional vector database for embedding-based
similarity search, supporting both single-tenant and multi-tenant scenarios.
"""

# Import the main classes from the database module
from .dbs import NanoVectorDB, MultiTenantNanoVDB

# Package metadata
__version__ = "0.0.4.3"  # Current version of the package
__author__ = "Jianbai Ye"  # Package author
__url__ = "https://github.com/gusye1234/nano-vectordb"  # Project repository URL
