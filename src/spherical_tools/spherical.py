import numpy
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import geometric_slerp


def validate_coordinates(
    coords: ArrayLike,
    coordinate_system: str,
    dtype: type = numpy.float64,
) -> NDArray:
    """
    Validate that coords is an array_like and its last dimension matches:
      - exactly 3 if coordinate_system == 'cartesian'
      - 2 or 3 if coordinate_system in ('geographic','spherical')

    Parameters
    ----------
    coords : array_like
        The coordinates to validate.
    coordinate_system : str
        The coordinate system, one of 'cartesian', 'geographic', or 'spherical'.
    dtype : type, optional
        The data type to convert the coordinates to. Default is numpy.float64.

    Returns
    -------
    arr : ndarray
        the validated array

    Raises
    ------
    TypeError
        if coords isn’t array‑like
    ValueError
        if last dimension size is wrong for the chosen coordinate system
    TypeError
        if coords cannot be converted to the specified dtype
    """

    # normalize to ndarray
    arr = numpy.asarray(coords, dtype=dtype)

    # determine what's allowed
    if coordinate_system == "cartesian":
        allowed = {3}
    elif coordinate_system in ("geographic", "spherical"):
        allowed = {2, 3}
    elif coordinate_system == "polar":
        allowed = {2}
    else:
        raise ValueError(
            f"invalid coordinate_system={coordinate_system!r}; "
            "expected 'cartesian', 'geographic', 'spherical', or 'polar'"
        )

    # check that there's at least one dimension
    if arr.ndim == 0:
        raise ValueError(f"expected array with last dimension {allowed}, got scalar")

    last_dim = arr.shape[-1]
    if last_dim not in allowed:
        raise ValueError(
            f"for coordinate_system={coordinate_system!r}, "
            f"expected last dimension in {allowed}, got {last_dim}"
        )

    return arr


def cart2sph(cartesian_coord_array: ArrayLike, degrees: bool = False) -> NDArray:
    """
    Convert Cartesian coordinates to spherical coordinates.

    Takes a coordinate array in Cartesian form ``(x, y, z)`` and returns the corresponding
    spherical coordinates ``(r, θ, φ)``, where ``r`` is the radial distance, ``θ`` is the azimuthal
    angle in the XY-plane from the X-axis, and ``φ`` is the polar angle from the Z-axis.

    Parameters
    ----------
    cartesian_coord_array : array_like, shape (..., 3)
        Input array of Cartesian coordinates. Can be a single 3-element vector or
        an array of shape (N, 3).
    degrees : bool, optional
        If True, the output angles are returned in degrees. Default is False (radians).

    Returns
    -------
    spherical_coord_array : ndarray, shape (..., 3)
        Output array of spherical coordinates. Each row contains ``(r, θ, φ)``.

    Notes
    -----
    Based on: https://stackoverflow.com/a/4116899

    Examples
    --------
    >>> cart2sph([1, 0, 0])
    array([1.        , 0.        , 1.57079633])
    """

    # validate input coordinates
    cartesian_coord_array = validate_coordinates(
        cartesian_coord_array, coordinate_system="cartesian", dtype=numpy.float64
    )

    # create new array to hold spherical coordinates
    spherical_coord_array = numpy.empty(cartesian_coord_array.shape)

    # convert to spherical coordinates
    spherical_coord_array[..., 0] = numpy.linalg.norm(cartesian_coord_array, axis=-1)
    spherical_coord_array[..., 1] = numpy.arctan2(
        cartesian_coord_array[..., 1], cartesian_coord_array[..., 0]
    )
    spherical_coord_array[..., 2] = numpy.arccos(
        cartesian_coord_array[..., 2] / spherical_coord_array[..., 0]
    )

    # convert from radians to degrees if required, otherwise skip
    if degrees:
        spherical_coord_array[..., 1:] = numpy.rad2deg(spherical_coord_array[..., 1:])

    return spherical_coord_array


def sph2cart(spherical_coord_array: ArrayLike, degrees: bool = False) -> NDArray:
    """
    Convert spherical coordinates to Cartesian coordinates.

    Takes spherical coordinates ``(r, θ, φ)`` and converts them to
    Cartesian ``(x, y, z)``, where ``θ`` is the azimuthal angle in the XY-plane
    from the X-axis, and ``φ`` is the polar angle from the Z-axis.

    Parameters
    ----------
    spherical_coord_array : array_like, shape (..., 3)
        Input array of spherical coordinates. Each row should contain
        ``(radius, azimuth, polar)``.
    degrees : bool, optional
        If True, input angles are assumed to be in degrees. Default is False (radians).

    Returns
    -------
    cartesian_coord_array : ndarray, shape (..., 3)
        Output array of Cartesian coordinates.

    Notes
    -----
    Equations from:
    https://en.wikipedia.org/wiki/List_of_common_coordinate_transformations#From_spherical_coordinates

    Examples
    --------
    >>> import numpy as np
    >>> np.set_printoptions(suppress=True)  # to avoid scientific notation due to floating point rounding
    >>> sph2cart([1, 0, np.pi/2])
    array([1., 0., 0.])
    """

    # validate input coordinates
    spherical_coord_array = validate_coordinates(
        spherical_coord_array, coordinate_system="cartesian", dtype=numpy.float64
    )

    # create new array to hold Cartesian coordinates
    cartesian_coord_array = numpy.empty(spherical_coord_array.shape)

    # convert from degrees to radians if required, otherwise skip
    if degrees:
        spherical_coord_array[..., 1:] = numpy.deg2rad(spherical_coord_array[..., 1:])

    # now the conversion to Cartesian coords
    cartesian_coord_array[..., 0] = (
        spherical_coord_array[..., 0]
        * numpy.cos(spherical_coord_array[..., 1])
        * numpy.sin(spherical_coord_array[..., 2])
    )
    cartesian_coord_array[..., 1] = (
        spherical_coord_array[..., 0]
        * numpy.sin(spherical_coord_array[..., 1])
        * numpy.sin(spherical_coord_array[..., 2])
    )
    cartesian_coord_array[..., 2] = spherical_coord_array[..., 0] * numpy.cos(
        spherical_coord_array[..., 2]
    )

    return cartesian_coord_array


def geo2sph(geographic_coord_array: ArrayLike, degrees: bool = False) -> NDArray:
    """
    Convert geographic coordinates ``([r], lon, lat)`` to spherical coordinates ``([r], azimuth, polar)``.

    Geographic coordinates use latitude and longitude on a sphere. This function
    transforms them into mathematical spherical coordinates, where:
    - Azimuth = longitude
    - Polar angle = 90 - latitude

    Parameters
    ----------
    geographical_coord_array : array_like, shape (..., 2) or (..., 3)
        Input array of geographic coordinates. Each row should be either
        ``(lon, lat)`` or ``(radius, lon, lat)``.
    degrees : bool, optional
        If True, output angles are returned in degrees. Default is False (radians).

    Returns
    -------
    spherical_coord_array : ndarray, shape (..., 2) or (..., 3)
        Output array of spherical coordinates ``(azimuth, polar)`` or ``(radius, azimuth, polar)``.

    Examples
    --------
    >>> geo2sph([150, -33], degrees=True)
    array([150., 123.])
    """

    # validate input coordinates
    geographic_coord_array = validate_coordinates(
        geographic_coord_array, coordinate_system="geographic", dtype=numpy.float64
    )

    # create new array to hold spherical coordinates
    spherical_coord_array = geographic_coord_array.copy()

    # reverse orientation of polar angle
    spherical_coord_array[..., -1] = 90 - spherical_coord_array[..., -1]

    # assume that outgoing spherical coordinates should be in radians, so convert from degrees by default
    if not degrees:
        spherical_coord_array[..., -2:] = numpy.deg2rad(spherical_coord_array[..., -2:])

    return spherical_coord_array


def sph2geo(spherical_coord_array: ArrayLike, degrees: bool = False) -> NDArray:
    """
    Convert spherical coordinates ``([r], azimuth, polar)`` to geographic coordinates ``([r], lon, lat)``.

    This function takes spherical coordinates where:
    - Azimuth = longitude
    - Polar angle = angle from the Z-axis

    and returns geographic coordinates where:
    - Latitude = 90° - polar angle
    - Longitude = azimuth

    Parameters
    ----------
    spherical_coord_array : array_like, shape (..., 2) or (..., 3)
        Input array of spherical coordinates. Each row should be either
        ``(azimuth, polar)`` or ``(radius, azimuth, polar)``.
    degrees : bool, optional
        If True, output angles are returned in degrees. Default is False (radians).

    Returns
    -------
    geographical_coord_array : ndarray, shape (..., 2) or (..., 3)
        Output array in the form ``(lon, lat)`` or ``(radius, lon, lat)``.

    Examples
    --------
    >>> sph2geo([150, 60], degrees=True)
    array([150., 30.])
    """

    # validate input coordinates
    spherical_coord_array = validate_coordinates(
        spherical_coord_array, coordinate_system="spherical", dtype=numpy.float64
    )

    # create a new array to hold the geographical coordinates
    geographical_coord_array = spherical_coord_array.copy()

    # incoming spherical coordinates are assumed to be in radians, so convert to degrees by default
    if not degrees:
        geographical_coord_array[..., -2:] = numpy.rad2deg(
            geographical_coord_array[..., -2:]
        )
    # reverse orientation of polar angle
    geographical_coord_array[..., -1] = 90 - geographical_coord_array[..., -1]

    return geographical_coord_array


def cart2polar(cartesian_coord_array: ArrayLike, degrees: bool = False) -> NDArray:
    """
    Convert 2D Cartesian coordinates to polar coordinates.

    Takes Cartesian coordinates ``(x, y)`` and returns polar coordinates ``(r, θ)``,
    where ``r`` is the distance from the origin and ``θ`` is the angle from the X-axis.

    Parameters
    ----------
    cartesian_coord_array : array_like, shape (..., 2)
        Input array of 2D Cartesian coordinates.
    degrees : bool, optional
        If True, the output angle is in degrees. Default is False (radians).

    Returns
    -------
    polar_coord_array : ndarray, shape (..., 2)
        Output array of polar coordinates ``(r, θ)``.

    Examples
    --------
    >>> cart2polar([1, 1], degrees=True)
    array([1.41421356, 45.        ])
    """

    # validate input coordinates
    cartesian_coord_array = validate_coordinates(
        cartesian_coord_array, coordinate_system="polar", dtype=numpy.float64
    )

    # create new array to hold spherical coordinates
    polar_coord_array = numpy.empty(cartesian_coord_array.shape)

    # convert to spherical coordinates
    polar_coord_array[..., 0] = numpy.linalg.norm(cartesian_coord_array, axis=-1)
    polar_coord_array[..., 1] = numpy.arctan2(
        cartesian_coord_array[..., 1], cartesian_coord_array[..., 0]
    )

    # convert from radians to degrees if required, otherwise skip
    if degrees:
        polar_coord_array[..., 1] = numpy.rad2deg(polar_coord_array[..., 1])

    return polar_coord_array


def polar2cart(polar_coord_array: ArrayLike, degrees: bool = False) -> NDArray:
    """
    Convert polar coordinates to 2D Cartesian coordinates.

    Takes polar coordinates `(r, θ)` and returns Cartesian coordinates `(x, y)`.

    Parameters
    ----------
    polar_coord_array : array_like, shape (..., 2)
        Input array of polar coordinates.
    degrees : bool, optional
        If True, input angles are assumed to be in degrees. Default is False (radians).

    Returns
    -------
    cartesian_coord_array : ndarray, shape (..., 2)
        Output array of Cartesian coordinates.

    Examples
    --------
    >>> polar2cart([1, 45], degrees=True)
    array([0.70710678, 0.70710678])
    """

    # validate input coordinates
    polar_coord_array = validate_coordinates(
        polar_coord_array, coordinate_system="polar", dtype=numpy.float64
    )

    # convert from degrees to radians if required, otherwise skip
    if degrees:
        polar_coord_array[..., 1] = numpy.deg2rad(polar_coord_array[..., 1])

    # create new array to hold spherical coordinates
    cartesian_coord_array = numpy.empty(polar_coord_array.shape)

    # convert to spherical coordinates
    cartesian_coord_array[..., 0] = (
        numpy.cos(polar_coord_array[..., 1]) * polar_coord_array[..., 0]
    )
    cartesian_coord_array[..., 1] = (
        numpy.sin(polar_coord_array[..., 1]) * polar_coord_array[..., 0]
    )

    return cartesian_coord_array


def great_circle_distance(
    array_1: ArrayLike,
    array_2: ArrayLike,
    coordinate_system: str = "spherical",
    sphere_radius: float | ArrayLike | None = None,
) -> NDArray:
    """
    Compute great‐circle distances between points on a sphere using the haversine formula.

    Parameters
    ----------
    array_1 : ArrayLike
        First point or set of points. Shape ``(3,)`` or ``(N, 3)``.
        The format of the three values depends on ``coordinate_system``:
        - ``"spherical"``: ``(θ, φ)`` or ``(r, θ, φ)``, where ``θ`` is the azimuthal angle (longitude) and ``φ`` is the polar angle (colatitude).
        - ``"geographic"``: ``(lon, lat)`` or ``(r, lon, lat)``.
        - ``"cartesian"``: ``(x, y, z)``.
    array_2 : ArrayLike
        Second point or set of points. Must have the same shape and coordinate convention as ``array_1``.
    coordinate_system : {'spherical', 'geographic', 'cartesian'}, default: 'spherical'
        Coordinate system of the inputs. If ``'cartesian'``, inputs are converted internally with ``cart2sph``.
    sphere_radius : float or ArrayLike or None, default: None
        Radius of the sphere to use for all points (scalar) or one radius per point (shape ``(N,)``).
        If ``None``, the radii are taken from pairwise mean of the radii of the input arrays.

    Returns
    -------
    float or ndarray
        Great‐circle distance(s) in the same units as ``sphere_radius``. Returns a scalar if the inputs
        are 1D, otherwise an array of shape ``(N,)``.

    Notes
    -----
    The haversine formulation is numerically stable for small angles compared to the arc‐cosine method.
    All angles must be provided in radians. Supplying mixed or degree units will yield incorrect results.

    Raises
    ------
    AssertionError
        If ``coordinate_system`` is not one of ``{'spherical', 'cartesian'}``.
    ValueError
        If ``array_1`` and ``array_2`` cannot be broadcast to the same shape or have incompatible last dimensions.

    See Also
    --------
    numpy.arcsin, numpy.cos, numpy.sin

    References
    ----------
    .. [1] Haversine formula — https://en.wikipedia.org/wiki/Haversine_formula

    Examples
    --------
    Distances between two single points (1 radian apart in longitude on a unit sphere):
    >>> import numpy as np
    >>> a1 = [1.0, 0.0, np.pi / 2]  # (r, θ, φ)
    >>> a2 = [1.0, 1.0, np.pi / 2]
    >>> great_circle_distance(a1, a2)
    np.float64(0.9999999999999999)

    Vectorized over many points:
    >>> pts1 = np.array([[1., 0, np.pi / 2],
    ...                  [1., 0.5, 0.3]])
    >>> pts2 = np.array([[1., 1., np.pi / 2],
    ...                  [1., 0.7, 0.1]])
    >>> np.set_printoptions(suppress=True)  # avoid scientific notation for tiny round-off
    >>> great_circle_distance(pts1, pts2)
    array([1.        , 0.20293885])

    Override the radius (e.g., Earth mean radius in km):
    >>> R_earth = 6371.0
    >>> great_circle_distance(a1, a2, sphere_radius=R_earth)
    np.float64(6370.999999999999)
    """

    # check that the coordinate system is valid
    allowed_coordinate_systems = {"spherical", "geographic", "cartesian"}
    if not isinstance(coordinate_system, str):
        raise TypeError(
            f"great_circle_distance() expected a str for coordinates, got {type(coordinates).__name__!r}"
        )
    if coordinate_system not in allowed_coordinate_systems:
        raise ValueError(
            f"great_circle_distance() got invalid coordinates={coordinate_system!r}; "
            f"expected one of: {sorted(allowed_coordinate_systems)!r}"
        )

    # validate input coordinates for the specified coordinate system
    array_1 = validate_coordinates(
        array_1, coordinate_system=coordinate_system, dtype=numpy.float64
    )
    array_2 = validate_coordinates(
        array_2, coordinate_system=coordinate_system, dtype=numpy.float64
    )

    # check that the arrays can be broadcast to the same shape
    try:
        numpy.broadcast_shapes(array_1.shape, array_2.shape)
    except ValueError as e:
        raise ValueError(
            f"great_circle_distance() arguments could not be broadcast together: "
            f"{array_1.shape} vs {array_2.shape}"
        ) from e

    if not sphere_radius:
        if coordinate_system == "cartesian":
            # convert Cartesian coordinates to spherical coordinates
            array_1 = cart2sph(array_1)
            array_2 = cart2sph(array_2)
        elif coordinate_system in {"geographic", "spherical"}:
            if coordinate_system == "geographic":
                # convert geographic coordinates to spherical coordinates
                array_1 = geo2sph(array_1)
                array_2 = geo2sph(array_2)
            # if sphere_radius is not provided, caluclate the pairwise mean radii
            if array_1.shape[-1] == 2:
                raise ValueError(
                    f"great_circle_distance() requires 'sphere_radius' when using 'spherical' or 'geographic' coordinate systems for input arrays with last dimension 2; "
                    "provide 'sphere_radius' or use shape (3,) or (...,3) instead."
                )
        sphere_radius = (array_1[..., 0] + array_2[..., 0]) / 2.0
    else:
        sphere_radius = numpy.asarray(sphere_radius, dtype=numpy.float64)

    theta_1 = array_1[..., -2]
    theta_2 = array_2[..., -2]
    phi_1 = array_1[..., -1]
    phi_2 = array_2[..., -1]

    spherical_distance = (
        2.0
        * sphere_radius
        * numpy.arcsin(
            numpy.sqrt(
                (
                    1
                    - numpy.cos(phi_2 - phi_1)
                    + numpy.sin(phi_1)
                    * numpy.sin(phi_2)
                    * (1 - numpy.cos(theta_2 - theta_1))
                )
                / 2.0
            )
        )
    )

    return spherical_distance


def fill_great_circle(
    g0: ArrayLike,
    g1: ArrayLike,
    res: float = 1.0,
    n_points: int | None = None,
    return_angle: bool = False,
):
    """
    Sample points along the great-circle path between two geographic coordinates.

    parameters
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

    returns
    -------
    g_profile : (n_points, 2) ndarray
        Longitudes (deg) and latitudes (deg) along the great circle. Longitudes are
        unwrapped to avoid jumps at 180.
    angle_deg : float, optional
        Great-circle angle in degrees (only if ``return_angle`` is True).
    """
    # combine input coords into (2, 2) array
    geo = numpy.array([g0, g1], dtype=float)

    # prepend radius = 1 for each point (required by geo2sph/scipy slerp workflow)
    radii = numpy.ones((geo.shape[0], 1))
    geo_with_r = numpy.concatenate((radii, geo), axis=1)

    # conversions
    sph = geo2sph(geo_with_r)  # -> (r, theta, phi)
    c0, c1 = sph2cart(sph)  # two 3D cartesian vectors

    # angular separation in degrees
    angle_deg = numpy.degrees(numpy.arccos(numpy.dot(c0, c1)))

    # choose number of samples
    if n_points is None:
        n_points = int(numpy.ceil(angle_deg / res)) + 1  # +1 to include both endpoints

    # interpolate on the unit sphere
    t = numpy.linspace(0.0, 1.0, n_points)
    cart_profile = geometric_slerp(c0, c1, t=t)

    # back to geographic lon/lat
    sph_profile = cart2sph(cart_profile)
    geo_profile = sph2geo(sph_profile[:, 1:])  # drop radius

    # unwrap longitudes to avoid jumps across the dateline
    geo_profile[:, 0] = numpy.unwrap(
        numpy.radians(geo_profile[:, 0]), period=2 * numpy.pi
    )
    geo_profile[:, 0] = numpy.degrees(geo_profile[:, 0])

    if return_angle:
        return geo_profile, angle_deg
    return geo_profile
