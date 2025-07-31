import numpy as np
from numpy.typing import ArrayLike, NDArray
from .._core import _unit_sphere_angle


def great_circle_distance(
    arr1: ArrayLike,
    arr2: ArrayLike,
    *,
    degrees: bool = False,
    radius: float | ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Calculate the great-circle distance between two points on a sphere.

    The great-circle distance is the shortest distance between two points on the
    surface of a sphere, measured along its surface. If ``radius`` is not
    provided, it defaults to the unit sphere (radius = 1). In other words, the
    angle(s) between the points is returned. The input coordinates should be in
    spherical coordinates ``(θ, φ)``, where ``θ`` is the azimuthal angle from
    the X-axis in the XY-plane, and ``φ`` is the polar angle from the Z-axis.

    Parameters
    ----------
    arr1 : array_like, shape (..., 2)
        First point or sequence of points in spherical coordinates (θ, φ).
    arr2 : array_like, shape (..., 2)
        Second point or sequence of points in spherical coordinates (θ, φ).
    degrees : bool, optional
        If True, the output is in degrees. Default is False (radians).
    radius : float or array_like, optional
        Radius of the sphere. If None, defaults to 1 (unit sphere).

    Returns
    -------
    ndarray, shape (...,)
        Great-circle distance(s) between the points in radians or degrees.
    """

    return
