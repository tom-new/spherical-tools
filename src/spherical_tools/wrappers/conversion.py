import numpy as np
from numpy.typing import ArrayLike, NDArray
from .._core import (
    _cart2sph,
    _sph2cart,
    _sph2geo2,
    _sph2geo3,
    _geo2sph2,
    _geo2sph3,
    _cart2geo,
    _geo2cart,
    _cart2polar,
    _polar2cart,
)
from .decorators import ensure_coords


@ensure_coords(ndim=3, name_in="Cartesian", name_out="spherical", convert_output=True)
def cart2sph(arr: ArrayLike) -> NDArray[np.float64]:
    """Convert Cartesian coordinates to spherical coordinates.

    Takes a coordinate array in Cartesian form ``(x, y, z)`` and returns the
    corresponding spherical coordinates ``(r, θ, φ)``, where ``r`` is the
    radial distance, ``θ`` is the azimuthal angle from the X-axis in the
    XY-plane, and ``φ`` is the polar angle from the Z-axis.

    Parameters
    ----------
    arr : array_like, shape (..., 3)
        Input array of Cartesian coordinates. The last dimension should be
        ``(x, y, z)``.
    degrees : bool, optional
        If True, the output angles are returned in degrees. Default is
        False (radians).

    Returns
    -------
    out : ndarray, shape (..., 3)
        Output array of spherical coordinates. The last dimension is
        ``(r, θ, φ)``.

    Notes
    -----
    Based on: https://stackoverflow.com/a/4116899

    Examples
    --------
    >>> cart2sph([1, 0, 0])
    array([1.        , 0.        , 1.57079633])
    """

    return _cart2sph(arr)


@ensure_coords(ndim=3, name_in="spherical", name_out="Cartesian", convert_input=True)
def sph2cart(arr: ArrayLike) -> NDArray[np.float64]:
    """Convert spherical coordinates to Cartesian coordinates.

    Takes a coordinate array in spherical coordinates ``(r, θ, φ)`` and returns
    the corresponding Cartesian coordinates ``(x, y, z)``, where ``r`` is the
    radial distance, ``θ`` is the azimuthal angle from the X-axis in the XY-
    plane, and ``φ`` is the polar angle from  the Z-axis.

    Parameters
    ----------
    arr : array_like, shape (..., 3)
        Input array of spherical coordinates. The last dimenstion should
        be ``(r, θ, φ)``.
    degrees : bool, optional
        If True, input angles are assumed to be in degrees. Default is False
        (radians).

    Returns
    -------
    out : ndarray, shape (..., 3)
        Output array of Cartesian coordinates. The last dimension is
        ``(x, y, z)``.

    Notes
    -----
    Equations from:
    https://en.wikipedia.org/wiki/List_of_common_coordinate_transformations#From_spherical_coordinates

    Examples
    --------
    >>> import numpy as np
    >>> # avoid scientific notation due to floating point rounding
    >>> np.set_printoptions(suppress=True)
    >>> sph2cart([1, 0, np.pi/2])
    array([1., 0., 0.])
    """

    return _sph2cart(arr)


@ensure_coords(
    ndim=(2, 3),
    name_in="geographic",
    name_out="spherical",
    convert_input=True,
    convert_output=True,
)
def geo2sph(arr: ArrayLike) -> NDArray[np.float64]:
    """Convert geographic coordinates to spherical coordinates.

    Takes a coordinate array in geographic coordinates ``([r,] lon, lat)`` and
    returns the corresponding spherical coordinates ``([r,] θ, φ)``, where
    ``r`` is the radial distance, ``θ`` is the azimuthal angle from the X-axis
    in the XY-plane, and ``φ`` is the polar angle from the Z-axis.

    Parameters
    ----------
    arr : array_like, shape (..., 2) or (..., 3)
        Input array of geographic coordinates. The last dimension should be
        either ``(lon, lat)`` or ``(r, lon, lat)``.
    degrees : bool, optional
        If True, input and output angles are in degrees. Default is False
        (radians).

    Returns
    -------
    out : ndarray, shape (..., 2) or (..., 3)
        Output array of spherical coordinates. The last dimension is ``(θ, φ)``
        or ``(r, θ, φ)``.

    Examples
    --------
    >>> geo2sph([150, -33], degrees=True)
    array([150., 123.])
    """

    if arr.shape[-1] == 2:
        return _geo2sph2(arr)
    elif arr.shape[-1] == 3:
        return _geo2sph3(arr)


@ensure_coords(
    ndim=(2, 3),
    name_in="spherical",
    name_out="geographic",
    convert_input=True,
    convert_output=True,
)
def sph2geo(arr: ArrayLike) -> NDArray[np.float64]:
    """Convert spherical coordinates to geographic coordinates.

    Takes a coordinate array in spherical coordinates ``([r,] θ, φ)``, where
    ``r`` is the radial distance, ``θ`` is the azimuthal angle from the X-axis
    in the XY-plane, and ``φ`` is the polar angle from the Z-axis, and returns
    the corresponding geographic coordinates ``([r,] lon, lat)``.

    Parameters
    ----------
    arr : array_like, shape (..., 2) or (..., 3)
        Input array of spherical coordinates. The last dimension should be
        either ``(θ, φ)`` or ``(r, θ, φ)``.
    degrees : bool, optional
        If True, input and output angles are in degrees. Default is False
        (radians).

    Returns
    -------
    out : ndarray, shape (..., 2) or (..., 3)
        Output array of geographic coordinates. The last dimension is either
        ``(lon, lat)`` or ``(r, lon, lat)``.

    Examples
    --------
    >>> sph2geo([150, 60], degrees=True)
    array([150., 30.])
    """

    if arr.shape[-1] == 2:
        return _sph2geo2(arr)
    elif arr.shape[-1] == 3:
        return _sph2geo3(arr)


@ensure_coords(ndim=3, name_in="Cartesian", name_out="geographic", convert_output=True)
def cart2geo(arr: ArrayLike) -> NDArray[np.float64]:
    """Convert Cartesian coordinates to geographic coordinates.

    Takes a coordinate array in Cartesian form ``(x, y, z)`` and returns the
    corresponding geographic coordinates ``(r, lon, lat)``.

    Parameters
    ----------
    arr : array_like, shape (..., 3)
        Input array of Cartesian coordinates. The last dimension should be
        ``(x, y, z)``.
    degrees : bool, optional
        If True, output angles are returned in degrees. Default is False
        (radians).

    Returns
    -------
    out : ndarray, shape (..., 3)
        Output array of geographic coordinates. The last dimension is
        ``(r, lon, lat)``.

    Examples
    --------
    >>> cart2geo([1, 0, 0])
    array([1.        , 0.        , 0.        ])
    """

    return _cart2geo(arr)


@ensure_coords(ndim=3, name_in="geographic", name_out="Cartesian", convert_input=True)
def geo2cart(arr: ArrayLike) -> NDArray[np.float64]:
    """Convert geographic coordinates to Cartesian coordinates.

    Takes a coordinate array in geographic form ``(r, lon, lat)`` and returns
    the corresponding Cartesian coordinates ``(x, y, z)``.

    Parameters
    ----------
    arr : array_like, shape (..., 3)
        Input array of geographic coordinates. The last dimension should be
        ``(r, lon, lat)``.
    degrees : bool, optional
        If True, input angles are assumed to be in degrees. Default is False
        (radians).

    Returns
    -------
    out : ndarray, shape (..., 3)
        Output array of Cartesian coordinates. The last dimension is
        ``(x, y, z)``.

    Examples
    --------
    >>> geo2cart([1, 0, 0])
    array([1., 0., 0.])
    """

    return _geo2cart(arr)


@ensure_coords(ndim=2, name_in="Cartesian", name_out="polar", convert_output=True)
def cart2polar(arr: ArrayLike) -> NDArray[np.float64]:
    """Convert Cartesian coordinates to polar coordinates.

    Takes a coordinate array in Cartesian form ``(x, y)`` and returns the
    corresponding polar coordinates ``(r, θ)``, where ``r`` is the radial
    distance and ``θ`` is the azimuthal angle from the X-axis.

    Parameters
    ----------
    arr : array_like, shape (..., 2)
        Input array of Cartesian coordinates. The last dimension should be
        ``(x, y)``.
    degrees : bool, optional
        If True, output angles are returned in degrees. Default is False
        (radians).

    Returns
    -------
    out : ndarray, shape (..., 2)
        Output array of polar coordinates. The last dimension is ``(r, θ)``.

    Examples
    --------
    >>> cart2polar([1, 0])
    array([1., 0.])
    """

    return _cart2polar(arr)


@ensure_coords(ndim=2, name_in="polar", name_out="Cartesian", convert_input=True)
def polar2cart(arr: ArrayLike) -> NDArray[np.float64]:
    """Convert polar coordinates to Cartesian coordinates.

    Takes a coordinate array in polar form ``(r, θ)`` and returns the
    corresponding Cartesian coordinates ``(x, y)``, where ``r`` is the radial
    distance and ``θ`` is the azimuthal angle from the X-axis.

    Parameters
    ----------
    arr : array_like, shape (..., 2)
        Input array of polar coordinates. The last dimension should be
        ``(r, θ)``.
    degrees : bool, optional
        If True, input angles are assumed to be in degrees. Default is False
        (radians).

    Returns
    -------
    out : ndarray, shape (..., 2)
        Output array of Cartesian coordinates. The last dimension is ``(x, y)``.

    Examples
    --------
    >>> polar2cart([1, 0])
    array([1., 0.])
    """

    return _polar2cart(arr)
