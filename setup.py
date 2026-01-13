"""Legacy setup.py for backwards compatibility. Delegates to pyproject.toml."""
from skbuild import setup

if __name__ == "__main__":
    setup()
