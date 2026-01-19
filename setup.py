"""
Legacy setup.py for backwards compatibility.

Modern builds should use pyproject.toml with scikit-build-core:
    pip install .

This file exists only for compatibility with tools that expect setup.py.
"""

# For scikit-build-core, we don't need anything here
# The build backend is specified in pyproject.toml
# Just importing this will work for legacy tools
import sys

if __name__ == "__main__":
    sys.stderr.write(
        "Error: setup.py is deprecated. Use 'pip install .' instead.\n"
        "Build configuration is in pyproject.toml.\n"
    )
    sys.exit(1)

