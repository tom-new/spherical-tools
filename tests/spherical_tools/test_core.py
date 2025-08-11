import math
import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose

# import from the package layout; adjust if your package name/path is different
from spherical_tools._core import (
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
    _unit_sphere_angle,
)


def rand_sph(n, *, rng=None):
    """Make random spherical coords (r, azimuth theta, colatitude phi).

    Returns
    -------
    arr : (n, 3) np.ndarray
    """
    rng = np.random.default_rng(rng)
    r = rng.uniform(1e-6, 10.0, size=n)  # avoid r == 0 to keep angles well-defined
    theta = rng.uniform(-np.pi, np.pi, size=n)  # azimuth
    phi = rng.uniform(0.0, np.pi, size=n)  # colatitude
    return np.stack([r, theta, phi], axis=-1)


def rand_geo(n, *, rng=None):
    """Make random geographic coords (r, lon, lat)."""
    rng = np.random.default_rng(rng)
    r = rng.uniform(1e-6, 10.0, size=n)
    lon = rng.uniform(-np.pi, np.pi, size=n)
    lat = rng.uniform(-np.pi / 2, np.pi / 2, size=n)
    return np.stack([r, lon, lat], axis=-1)


def test_cart2sph_known_axes():
    # x axis
    out = _cart2sph(np.array([1.0, 0.0, 0.0]))
    assert_allclose(out, np.array([1.0, 0.0, np.pi / 2]))
    # y axis
    out = _cart2sph(np.array([0.0, 1.0, 0.0]))
    assert_allclose(out, np.array([1.0, np.pi / 2, np.pi / 2]))
    # +z axis
    out = _cart2sph(np.array([0.0, 0.0, 1.0]))
    assert_allclose(out, np.array([1.0, 0.0, 0.0]))
    # -z axis
    out = _cart2sph(np.array([0.0, 0.0, -1.0]))
    assert_allclose(out, np.array([1.0, 0.0, np.pi]))


def test_cart2sph_zero_vector_no_warning_and_nan_polar():
    x = np.array([0.0, 0.0, 0.0])
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = _cart2sph(x)
    assert len(rec) == 0  # no runtime warnings
    assert out[0] == 0.0  # radius
    assert math.isnan(out[2])  # polar/colatitude is undefined => nan
    assert out[1] == 0.0  # azimuth from arctan2(0,0) is 0 by convention


def test_sph2cart_known_cases():
    # r=1, theta=0, phi=pi/2 => x=1,y=0,z=0
    out = _sph2cart(np.array([1.0, 0.0, np.pi / 2]))
    assert_allclose(out, np.array([1.0, 0.0, 0.0]), atol=1e-15)
    # r=1, theta=pi/2, phi=pi/2 => x=0,y=1,z=0
    out = _sph2cart(np.array([1.0, np.pi / 2, np.pi / 2]))
    assert_allclose(out, np.array([0.0, 1.0, 0.0]), atol=1e-15)
    # r=1, theta=any, phi=0 => x=0,y=0,z=1
    out = _sph2cart(np.array([1.0, 1.234, 0.0]))
    assert_allclose(out, np.array([0.0, 0.0, 1.0]), atol=1e-15)


@pytest.mark.parametrize("shape", [(10, 3), (4, 5, 3)])
def test_cart_sph_roundtrip_random(shape):
    # make random spherical, convert to cartesian and back
    n = np.prod(shape[:-1])
    sph = rand_sph(n, rng=0).reshape(shape)
    cart = _sph2cart(sph)
    sph_rt = _cart2sph(cart)

    # when phi is 0 or pi, azimuth is irrelevant; mask those
    phi = sph[..., 2]
    mask = (phi > 1e-12) & (phi < np.pi - 1e-12)

    assert_allclose(sph_rt[..., 0], sph[..., 0])  # radius
    assert_allclose(sph_rt[..., 1][mask], sph[..., 1][mask])
    assert_allclose(sph_rt[..., 2], sph[..., 2])


def test_sph_geo_two_and_three_component_conversions():
    # 2-component: (lon=theta, lat=pi/2 - phi)
    sph2 = np.array([np.pi / 3, np.pi / 6])  # theta=60°, phi=30°
    geo2 = _sph2geo2(sph2)
    assert_allclose(geo2, np.array([np.pi / 3, np.pi / 2 - np.pi / 6]))
    sph2_rt = _geo2sph2(geo2)
    assert_allclose(sph2_rt, sph2)

    # 3-component keeps radius intact
    sph3 = np.array([2.0, -1.0, 0.7])
    geo3 = _sph2geo3(sph3)
    assert_allclose(geo3, np.array([2.0, -1.0, np.pi / 2 - 0.7]))
    sph3_rt = _geo2sph3(geo3)
    assert_allclose(sph3_rt, sph3)


def test_cart2geo_known_axes_and_zero_vector():
    # +z => lat=+pi/2
    out = _cart2geo(np.array([0.0, 0.0, 1.0]))
    assert_allclose(out, np.array([1.0, 0.0, np.pi / 2]))
    # x axis => lat=0, lon=0
    out = _cart2geo(np.array([1.0, 0.0, 0.0]))
    assert_allclose(out, np.array([1.0, 0.0, 0.0]))
    # zero vector: radius 0, lon 0, lat nan (arcsin(0/0))
    out = _cart2geo(np.array([0.0, 0.0, 0.0]))
    assert out[0] == 0.0
    assert out[1] == 0.0
    assert math.isnan(out[2])


@pytest.mark.parametrize("shape", [(20, 3), (3, 7, 3)])
def test_geo_cart_roundtrip_random(shape):
    n = np.prod(shape[:-1])
    geo = rand_geo(n, rng=1).reshape(shape)
    cart = _geo2cart(geo)
    geo_rt = _cart2geo(cart)

    # when |lat| == pi/2, lon is irrelevant; mask poles
    lat = geo[..., 2]
    mask = np.abs(lat) < (np.pi / 2 - 1e-12)

    assert_allclose(geo_rt[..., 0], geo[..., 0])  # radius
    assert_allclose(geo_rt[..., 1][mask], geo[..., 1][mask])
    assert_allclose(geo_rt[..., 2], geo[..., 2])


@pytest.mark.parametrize("shape", [(15, 2), (6, 5, 2)])
def test_polar_cart_roundtrip_random(shape):
    rng = np.random.default_rng(2)
    n = np.prod(shape[:-1])
    r = rng.uniform(0.0, 10.0, size=n)
    ang = rng.uniform(-np.pi, np.pi, size=n)
    polar = np.stack([r, ang], axis=-1).reshape(shape)

    cart = _polar2cart(polar)
    polar_rt = _cart2polar(cart)

    # when r == 0, angle is irrelevant; mask zeros
    mask = polar[..., 0] > 1e-12

    assert_allclose(polar_rt[..., 0], polar[..., 0])
    assert_allclose(polar_rt[..., 1][mask], polar[..., 1][mask])


def test_unit_sphere_angle_known_cases():
    # same point (theta, phi)
    a = np.array([0.0, np.pi / 2])  # on equator at theta=0
    assert _unit_sphere_angle(a, a) == pytest.approx(0.0)

    # 90° apart along equator => phi constant = pi/2, delta-theta = pi/2
    b = np.array([np.pi / 2, np.pi / 2])
    ang = _unit_sphere_angle(a, b)
    assert ang == pytest.approx(np.pi / 2)

    # antipodal on equator => delta-theta = pi
    c = np.array([np.pi, np.pi / 2])
    ang = _unit_sphere_angle(a, c)
    assert ang == pytest.approx(np.pi)

    # vectorized broadcasting along equator
    thetas = np.array([0.0, np.pi / 2, np.pi])
    phis = np.full_like(thetas, np.pi / 2)
    arr1 = np.stack([thetas, phis], axis=-1)
    arr2 = np.array([0.0, np.pi / 2])  # reference: theta=0 on equator
    out = _unit_sphere_angle(arr1, arr2)
    assert_allclose(out, np.array([0.0, np.pi / 2, np.pi]))

    assert_allclose(out, np.array([0.0, np.pi / 2, np.pi]))
