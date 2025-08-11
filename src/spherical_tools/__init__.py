from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("spherical_tools")
except PackageNotFoundError:
    __version__ = "unknown"

# pull in child packages/modules first so their __all__ exist
from . import wrappers as _wrappers

# re-export only what those modules declare as public
from .wrappers import *

# package-level public API for `spherical_tools`
__all__ = _wrappers.__all__
