from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("spherical_tools")
except PackageNotFoundError:
    __version__ = "unknown"

from .wrappers.conversion import (
    cart2sph,
    sph2cart,
    geo2sph,
    sph2geo,
    cart2polar,
    polar2cart,
)

from .wrappers.geodetic import (
    great_circle_distance,
    crosses_dateline,
    fill_great_circle,
)

__all__ = [
    "cart2sph",
    "sph2cart",
    "geo2sph",
    "sph2geo",
    "cart2polar",
    "polar2cart",
    "great_circle_distance",
    "crosses_dateline",
    "fill_great_circle",
]
