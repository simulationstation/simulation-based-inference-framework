"""
Analysis Pack Standard (APS) - Core module for loading and validating analysis packs.

This module provides the standard interface for working with reusable statistical
models of collider measurements and searches.
"""

from .loaders import load, validate_pack
from .schemas import METADATA_SCHEMA, LIKELIHOOD_SCHEMA, CONSTRAINTS_SCHEMA

__version__ = "0.1.0"
__all__ = ["load", "validate_pack", "METADATA_SCHEMA", "LIKELIHOOD_SCHEMA", "CONSTRAINTS_SCHEMA"]
