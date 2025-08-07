import warnings
import numpy as np
from numpy.typing import ArrayLike, NDArray
from .._core import _unit_sphere_angle
from .._core import _geo2cart, _cart2geo, _cart2sph
from .._vendor.slerp import _geometric_slerp
from .decorators import ensure_units


def great_circle_distance(
    arr1: ArrayLike,
    arr2: ArrayLike,
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

    angle = _unit_sphere_angle(arr1, arr2)
    return angle if radius is None else angle * np.asarray(radius, dtype=np.float64)


def crosses_dateline(
    arr1: ArrayLike,
    arr2: ArrayLike,
) -> NDArray[np.bool_]:
    """Check if the great-circle path between two points crosses the dateline.

    The dateline is considered crossed if the azimuthal angle ``θ`` of the two
    points differs by more than 180 degrees.

    Parameters
    ----------
    arr1 : array_like, shape (..., 2)
        First point or sequence of points in spherical coordinates (θ, φ).
    arr2 : array_like, shape (..., 2)
        Second point or sequence of points in spherical coordinates (θ, φ).

    Returns
    -------
    ndarray, shape (...,)
        Boolean array indicating whether the great-circle path crosses the dateline.
    """

    # Normalise the azimuthal angles to the range [-π, π)
    arr2[..., 0] = ((arr2[..., 0] + np.pi) % (2 * np.pi)) - np.pi
    arr1[..., 0] = ((arr1[..., 0] + np.pi) % (2 * np.pi)) - np.pi

    # Calculate the absolute difference in azimuthal angles
    angle_diff = np.abs(arr1[..., 0] - arr2[..., 0])

    return angle_diff > np.pi


def fill_great_circle(
    arr1: ArrayLike,
    arr2: ArrayLike,
    res: float = 1.0,
    n_points: int | None = None,
    return_angle: bool = False,
    tol: float = 1e-7,
):
    """
    Sample points along the great-circle path between two geographic coordinates.

    Parameters
    ----------
    g0, g1 : array-like
        (lon, lat) in degrees.
    res : float, default 1.0
        Target angular spacing in degrees between consecutive points. Ignored if
        ``n_points`` is given.
    n_points : int or None, default None
        Number of samples (including endpoints). If None, it is computed from ``res``.
    return_angle : bool, default False
        If True, also return the great-circle angle (degrees) between g0 and g1.
    tol : float
        The absolute tolerance for determining if arr1 and arr2 are antipodes.

    Returns
    -------
    g_profile : (n_points, 2) ndarray
        Longitudes (deg) and latitudes (deg) along the great circle. Longitudes are
        unwrapped to avoid jumps at 180.
    angle_deg : float, optional
        Great-circle angle in degrees (only if ``return_angle`` is True).
    """

    arr1 = np.asarray(arr1, dtype=np.float64)
    arr2 = np.asarray(arr2, dtype=np.float64)

    res = float(res)

    if n_points is not None:
        n_points = int(n_points)
        if n_points < 2:
            raise ValueError("n_points must be at least 2 to include both endpoints.")

    if arr1.ndim != 1 or arr2.ndim != 1:
        raise ValueError("Coordinate arrays arr1 and arr2 must be shape (2,)")

    if arr1.size != arr2.size:
        raise ValueError("The dimensions of arr1 and arr2 must match (have same size)")

    if np.array_equal(arr1, arr2):
        return np.linspace(arr1, arr1, t.size)

    arr1 = np.deg2rad(arr1)
    arr2 = np.deg2rad(arr2)

    radii = np.ones((arr1.shape[:-1] + (1,)), dtype=np.float64)
    arr1 = np.concatenate((radii, arr1), axis=-1)
    arr2 = np.concatenate((radii, arr2), axis=-1)

    arr1 = _geo2cart(arr1)
    arr2 = _geo2cart(arr2)

    # separation
    coord_dist = np.linalg.norm(arr2 - arr1, axis=-1)

    # diameter of 2 within tolerance means antipodes, which is a problem
    # for all unit n-spheres (even the 0-sphere would have an ambiguous path)
    if np.allclose(coord_dist, 2.0, rtol=0, atol=tol):
        warnings.warn(
            "start and end are antipodes "
            "using the specified tolerance; "
            "this may cause ambiguous slerp paths",
            stacklevel=2,
        )

    if not isinstance(tol, float):
        raise ValueError("tol must be a float")
    else:
        tol = np.fabs(tol)

    angular_distance = _unit_sphere_angle(_cart2sph(arr1), _cart2sph(arr2))
    print(angular_distance)
    # choose number of samples
    if n_points is None:
        n_points = (
            int(np.ceil(angular_distance / np.deg2rad(res))) + 1
        )  # +1 to include both endpoints

    # generate interpolation points
    t = np.linspace(0.0, 1.0, n_points)

    # interpolate on the unit sphere
    profile = _geometric_slerp(arr1, arr2, t)

    # convert to geographic coords and drop radius
    profile = _cart2geo(profile)[:, 1:]

    # unwrap longitudes to avoid jumps across the dateline
    profile[:, 0] = np.unwrap(profile[:, 0], period=2 * np.pi)

    # convert to degrees
    profile = np.rad2deg(profile)

    if return_angle:
        return profile, angular_distance
    return profile
