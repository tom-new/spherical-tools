__all__ = [
    "great_circle_distance",
    "crosses_dateline",
    "fill_great_circle",
]

import warnings
from typing import Literal, Tuple
import numpy as np
from numpy.typing import ArrayLike, NDArray
from .._core import _unit_sphere_angle
from .._core import _geo2cart, _cart2geo, _cart2sph, _geo2sph2, _sph2cart
from .._vendor.slerp import _geometric_slerp
from .decorators import ensure_units

CoordinateSystem = Literal["spherical", "geographic"]


def great_circle_distance(
    arr1: ArrayLike,
    arr2: ArrayLike,
    degrees: bool = False,
    radius: float | ArrayLike | None = None,
    *,
    coordinate_system: CoordinateSystem = "spherical",
) -> NDArray[np.float64]:
    """Great-circle distance (or central angle) between two points on a sphere.

    The great-circle distance is the shortest path along the sphere’s surface.
    If ``radius`` is ``None``, the function returns the central angle. In that
    angular-only case, ``degrees`` controls the output unit. If ``radius`` is
    provided, the result is the great circle distance in the same units as
    ``radius``, regardless of ``degrees``.

    Parameters
    ----------
    arr1, arr2 : array_like, shape (..., 2)
        Input coordinates. Interpretation depends on ``coordinate_system``:
        - ``'spherical'``: ``(θ, φ)`` where ``θ`` is azimuth in the XY-plane
          from +X, and ``φ`` is the polar (colatitude) angle from +Z.
        - ``'geographic'``: ``(lon, lat)`` with geodetic latitude.
        Inputs must be broadcastable against each other.
    degrees : bool, optional
        If ``True``, input angles are interpreted as degrees. When
        ``radius is None``, the returned central angle is also in degrees.
        Default is ``False`` (radians).
    radius : float or array_like, optional
        Sphere radius. If provided, the output is a distance (units of
        ``radius``). If ``None``, returns the central angle.
    coordinate_system : {'spherical', 'geographic'}, optional
        Coordinate system of the inputs. Default is ``'spherical'``.

    Returns
    -------
    ndarray of float64, shape (...,)
        Central angle (if ``radius`` is ``None``) or great-circle distance
        (if ``radius`` is provided).

    Notes
    -----
    - Internally converts inputs to spherical ``(θ, φ)`` in radians and calls a
      numerically stable haversine-style core.

    Examples
    --------
    >>> # lon/lat in degrees (Sydney to Perth); get angular separation (deg)
    >>> syd = [151.21, -33.87]
    >>> per = [115.86, -31.95]
    >>> ang_deg = great_circle_distance(syd, per, degrees=True, coordinate_system="geographic")
    >>> # now as a distance using Earth's mean radius (km)
    >>> R_earth_km = 6371.0088
    >>> d_km = great_circle_distance(syd, per, degrees=True, coordinate_system="geographic", radius=R_earth_km)
    """

    if coordinate_system not in ("geographic", "spherical"):
        raise ValueError("coordinate_system must be 'geographic' or 'spherical'")

    a1 = np.asarray(arr1, dtype=np.float64)
    a2 = np.asarray(arr2, dtype=np.float64)

    if a1.shape[-1] != 2 or a2.shape[-1] != 2:
        raise ValueError("a1 and a2 must have shape (..., 2)")

    a1, a2 = np.broadcast_arrays(a1, a2)

    if degrees:
        a1 = np.deg2rad(a1)
        a2 = np.deg2rad(a2)

    if coordinate_system == "geographic":
        a1 = _geo2sph2(a1)
        a2 = _geo2sph2(a2)

    angle_rad = _unit_sphere_angle(a1, a2)
    if radius is None:
        return np.rad2deg(angle_rad) if degrees else angle_rad
    return angle_rad * np.asarray(radius, dtype=np.float64)


def crosses_dateline(
    arr1: ArrayLike,
    arr2: ArrayLike,
    *,
    coordinate_system: CoordinateSystem = "spherical",
    degrees: bool = False,
) -> NDArray[np.bool_]:
    """
    Check whether the shortest great-circle path between two points crosses the
    antimeridian (dateline).

    The dateline is taken as longitudes ±π (±180°). The path is considered to
    cross if the absolute difference in azimuth/longitude between the two
    points, after normalizing each to ``[-π, π)``, exceeds π.

    Parameters
    ----------
    arr1, arr2 : array_like, shape (..., 2)
        Input coordinates. Interpretation depends on ``coordinate_system``:
        - ``'spherical'``: ``(θ, φ)`` where ``θ`` is azimuth in the XY-plane
          from +X, and ``φ`` is the polar (colatitude) angle from +Z.
        - ``'geographic'``: ``(lon, lat)`` with geodetic latitude.
        Inputs must be broadcastable against each other.
    coordinate_system : {'spherical', 'geographic'}, optional
        Coordinate system of the inputs. Default is ``'spherical'``.
    degrees : bool, optional
        If ``True``, interpret input angles as degrees. Default is ``False``
        (radians).

    Returns
    -------
    ndarray of bool, shape (...,)
        Boolean mask indicating whether the great-circle path crosses the
        dateline.

    Notes
    -----
    - Does not mutate the inputs.
    - Uses only the azimuth/longitude component.
    - Equality at exactly π is treated as *not* crossing.
    """
    a1 = np.asarray(arr1, dtype=np.float64)
    a2 = np.asarray(arr2, dtype=np.float64)

    if a1.shape[-1] != 2 or a2.shape[-1] != 2:
        raise ValueError("arr1 and arr2 must have shape (..., 2)")

    # broadcast once, up front
    a1, a2 = np.broadcast_arrays(a1, a2)

    # pick the azimuth/longitude component
    if coordinate_system in ["geographic", "spherical"]:
        th1 = a1[..., 0]
        th2 = a2[..., 0]
    else:
        raise ValueError("coordinate_system must be 'spherical' or 'geographic'")

    if degrees:
        th1 = np.deg2rad(th1)
        th2 = np.deg2rad(th2)

    # normalize each angle to [-π, π)
    th1 = (th1 + np.pi) % (2.0 * np.pi) - np.pi
    th2 = (th2 + np.pi) % (2.0 * np.pi) - np.pi

    angle_diff = np.abs(th1 - th2)
    return angle_diff > np.pi


def fill_great_circle(
    arr1: ArrayLike,
    arr2: ArrayLike,
    res: float = 1.0,
    n_points: int | None = None,
    return_angle: bool = False,
    tol: float = 1e-7,
    *,
    coordinate_system: CoordinateSystem = "geographic",
    degrees: bool = True,
) -> NDArray[np.float64] | Tuple[NDArray[np.float64], float]:
    """
    Sample points along the great-circle path between two points.

    Parameters
    ----------
    arr1, arr2
        Endpoints of the path, shape ``(2,)``. Interpretation depends on
        ``coordinate_system``:
          - ``'geographic'``: ``(lon, lat)``
          - ``'spherical'``: ``(θ, φ)`` where ``θ`` is azimuth in the XY-plane
            from +X, and ``φ`` is the polar (colatitude) angle from +Z.
    res
        Target angular spacing between consecutive samples. Interpreted in
        degrees if ``degrees=True`` and radians otherwise. Ignored if
        ``n_points`` is provided.
    n_points
        Number of samples, including both endpoints. If ``None``, computed as
        ``ceil(angle / res) + 1`` with a minimum of 2.
    return_angle
        If ``True``, also return the great-circle angle between the endpoints
        (in degrees if ``degrees=True``, else radians).
    tol
        Absolute tolerance to detect antipodal endpoints (ambiguous SLERP path).
    coordinate_system
        One of ``{'geographic', 'spherical'}``. Controls how coordinates are
        interpreted and how the output is expressed.
    degrees
        If ``True``, interpret inputs (and ``res``) as degrees and return the
        sampled coordinates in degrees. If ``False``, use radians throughout.

    Returns
    -------
    g_profile : (n_points, 2) ndarray of float64
        Coordinates along the great circle in the requested ``coordinate_system``
        and angle unit. For geographic, longitudes are unwrapped to avoid jumps
        at ±180° (or ±π). For spherical, azimuth ``θ`` is unwrapped.
    angle : float, optional
        Central angle between endpoints, returned only when
        ``return_angle=True`` (in degrees if ``degrees=True``, else radians).

    Notes
    -----
    - Uses geometric SLERP on unit vectors via core converters only:
      ``_geo2cart``, ``_sph2cart``, ``_cart2geo``, ``_cart2sph``,
      and ``_unit_sphere_angle``.
    """

    if coordinate_system not in ("geographic", "spherical"):
        raise ValueError("coordinate_system must be 'geographic' or 'spherical'")

    a1 = np.asarray(arr1, dtype=np.float64)
    a2 = np.asarray(arr2, dtype=np.float64)

    if a1.shape != (2,) or a2.shape != (2,):
        raise ValueError("arr1 and arr2 must both have shape (2,)")

    if n_points is not None:
        n_points = int(n_points)
        if n_points < 2:
            raise ValueError("n_points must be at least 2 to include both endpoints.")

    if not isinstance(res, (int, float)):
        raise ValueError("res must be a float-like value.")
    if not isinstance(tol, float):
        raise ValueError("tol must be a float.")

    # early-out for identical endpoints (preserve input units + convention)
    if np.allclose(a1, a2, rtol=0.0, atol=1e-15):
        n = 2 if n_points is None else n_points
        prof = np.repeat(a1[np.newaxis, :], n, axis=0)
        return (prof, 0.0 if not return_angle else (prof, 0.0))[
            0 if not return_angle else 1
        ]

    # convert inputs to radians for core converters
    a1_rad = np.deg2rad(a1) if degrees else a1
    a2_rad = np.deg2rad(a2) if degrees else a2

    # build unit cartesian vectors using core converters only
    if coordinate_system == "geographic":
        # geo (lon, lat) [rad] -> add r=1 -> cart -> sph -> drop radius
        rvec = np.array([1.0], dtype=np.float64)
        thphi1 = _geo2sph2(np.concatenate((rvec, a1_rad)))[1:]  # (θ, φ)
        thphi2 = _geo2sph2(np.concatenate((rvec, a2_rad)))[1:]  # (θ, φ)
    else:
        thphi1 = a1_rad  # already (θ, φ)
        thphi2 = a2_rad

    # compute the angle between the two points on the unit sphere
    angle_rad = _unit_sphere_angle(thphi1, thphi2)

    # detect antipodes via chord length ~ 2
    rvec = np.array([1.0], dtype=np.float64)
    p1 = _sph2cart(np.concatenate((rvec, thphi1)))  # (3,)
    p2 = _sph2cart(np.concatenate((rvec, thphi2)))  # (3,)
    chord = np.linalg.norm(p2 - p1)
    if np.allclose(chord, 2.0, rtol=0.0, atol=tol):
        warnings.warn(
            "start and end are antipodes using the specified tolerance; "
            "this may cause ambiguous SLERP paths",
            stacklevel=2,
        )

    # decide number of samples
    if n_points is None:
        if res <= 0:
            raise ValueError("res must be positive.")
        angle_unit = np.rad2deg(angle_rad) if degrees else angle_rad
        n_points = int(np.ceil(float(angle_unit) / float(res))) + 1
        n_points = max(n_points, 2)

    # interpolate on the unit sphere
    t = np.linspace(0.0, 1.0, n_points)
    profile_cart = _geometric_slerp(p1, p2, t)  # (n_points, 3)

    # convert back using core converters; unwrap periodic axis; then apply units
    conv = _cart2geo if coordinate_system == "geographic" else _cart2sph
    out = conv(profile_cart)[..., 1:]  # drop radius
    out[..., 0] = np.unwrap(out[..., 0], period=2.0 * np.pi)  # unwrap lon or θ
    if degrees:
        out = np.rad2deg(out)

    if return_angle:
        return out, (float(np.rad2deg(angle_rad)) if degrees else float(angle_rad))
    return out
