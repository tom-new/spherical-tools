import numpy as np
import pytest
from numpy.testing import assert_allclose

# import the module under test
from spherical_tools.wrappers import conversion as conv

RTOL = 1e-12
ATOL = 1e-12


def test_public_api_accepts_arraylikes():
    # ensures user-facing functions are friendly to lists
    import spherical_tools as st

    st.cart2polar([1, 0], degrees=True)
    st.cart2sph([1, 0, 0])


@pytest.mark.parametrize("degrees", [False, True])
def test_cart_sph_roundtrip(degrees):
    # generate random cartesian points away from the origin
    rng = np.random.default_rng(1234)
    cart = rng.normal(size=(100, 3))
    cart = cart / np.linalg.norm(cart, axis=-1, keepdims=True)  # unit sphere
    sph = conv.cart2sph(cart.copy(), degrees=degrees)
    cart_back = conv.sph2cart(sph.copy(), degrees=degrees)
    assert_allclose(cart_back, cart, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("degrees", [False, True])
def test_geo_cart_roundtrip(degrees):
    # sample random geographic lon/lat and radius=1
    rng = np.random.default_rng(5678)
    lon = rng.uniform(-np.pi, np.pi, size=200)
    lat = rng.uniform(-np.pi / 2, np.pi / 2, size=200)
    r = np.ones_like(lon)
    if degrees:
        lon = np.rad2deg(lon)
        lat = np.rad2deg(lat)
    geo = np.stack([r, lon, lat], axis=-1)
    expected = geo.copy()
    cart = conv.geo2cart(geo.copy(), degrees=degrees)
    geo_back = conv.cart2geo(cart.copy(), degrees=degrees)
    # when degrees=True, angles should compare in degrees; otherwise in radians
    assert_allclose(geo_back[..., 0], r, rtol=RTOL, atol=ATOL)  # radius
    assert_allclose(geo_back[..., 1:], expected[..., 1:], rtol=RTOL, atol=1e-9)


@pytest.mark.parametrize("use_radius", [False, True])
@pytest.mark.parametrize("degrees", [False, True])
def test_sph_geo_roundtrip(degrees, use_radius):
    rng = np.random.default_rng(42)
    theta = rng.uniform(-np.pi, np.pi, size=100)  # azimuth
    phi = rng.uniform(0, np.pi, size=100)  # polar
    if degrees:
        theta_in = np.rad2deg(theta)
        phi_in = np.rad2deg(phi)
    else:
        theta_in = theta
        phi_in = phi

    if use_radius:
        r = rng.uniform(0.5, 3.0, size=100)
        sph = np.stack([r, theta_in, phi_in], axis=-1)
        geo = conv.sph2geo(sph.copy(), degrees=degrees)
        expected = sph.copy()
        sph_back = conv.geo2sph(geo.copy(), degrees=degrees)
        assert_allclose(sph_back, expected, rtol=RTOL, atol=1e-9)
    else:
        sph = np.stack([theta_in, phi_in], axis=-1)
        geo = conv.sph2geo(sph.copy(), degrees=degrees)
        expected = sph.copy()
        sph_back = conv.geo2sph(geo.copy(), degrees=degrees)
        assert_allclose(sph_back, expected, rtol=RTOL, atol=1e-9)


def test_cart2geo_known_axes():
    # x-axis -> lon=0, lat=0
    out = conv.cart2geo(np.array([1.0, 0.0, 0.0]))
    assert_allclose(out, np.array([1.0, 0.0, 0.0]), rtol=RTOL, atol=ATOL)
    # y-axis -> lon=pi/2, lat=0
    out = conv.cart2geo(np.array([0.0, 1.0, 0.0]))
    assert_allclose(out, np.array([1.0, np.pi / 2, 0.0]), rtol=RTOL, atol=1e-12)
    # z-axis -> lat=pi/2
    out = conv.cart2geo(np.array([0.0, 0.0, 1.0]))
    assert_allclose(out, np.array([1.0, 0.0, np.pi / 2]), rtol=RTOL, atol=1e-12)
    # negative z-axis -> lat=-pi/2
    out = conv.cart2geo(np.array([0.0, 0.0, -1.0]))
    assert_allclose(out, np.array([1.0, 0.0, -np.pi / 2]), rtol=RTOL, atol=1e-12)


@pytest.mark.parametrize("degrees", [False, True])
def test_degrees_flag_behavior(degrees):
    cart = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    sph = conv.cart2sph(cart.copy(), degrees=degrees)
    if degrees:
        # angles should be in degrees
        assert_allclose(sph[1:], np.array([45.0, 54.73561032]), rtol=RTOL, atol=1e-6)
    else:
        assert_allclose(
            sph[1:],
            np.array([np.pi / 4, np.arccos(1 / np.sqrt(3))]),
            rtol=RTOL,
            atol=1e-12,
        )
    # roundtrip with degrees flag must succeed
    back = conv.sph2cart(sph, degrees=degrees)
    assert_allclose(back, cart, rtol=RTOL, atol=ATOL)


def test_shape_validation_errors():
    # wrong last-dim sizes should raise
    with pytest.raises(ValueError):
        conv.cart2sph(np.ones(4))
    with pytest.raises(ValueError):
        conv.sph2cart(np.ones(4))
    with pytest.raises(ValueError):
        conv.cart2geo(np.ones(4))
    with pytest.raises(ValueError):
        conv.geo2cart(np.ones(4))
    with pytest.raises(ValueError):
        conv.cart2polar(np.ones(3))
    with pytest.raises(ValueError):
        conv.polar2cart(np.ones(3))


def test_zero_vector_cart2sph():
    out = conv.cart2sph(np.array([0.0, 0.0, 0.0]))
    # radius==0, angles are ill-defined -> polar angle is nan
    assert out[0] == 0.0
    assert out[1] == 0.0
    assert np.isnan(out[2])


@pytest.mark.parametrize("degrees", [False, True])
def test_polar_roundtrip_and_axes(degrees):
    # axes
    assert_allclose(
        conv.cart2polar(np.array([1.0, 0.0]), degrees=degrees), np.array([1.0, 0.0])
    )
    assert_allclose(
        conv.cart2polar(np.array([0.0, 1.0]), degrees=degrees)[0],
        1.0,
        rtol=RTOL,
        atol=ATOL,
    )
    # roundtrip
    rng = np.random.default_rng(4321)
    r = rng.uniform(0.1, 5.0, size=50)
    theta = rng.uniform(-np.pi, np.pi, size=50)
    if degrees:
        theta = np.rad2deg(theta)
    pol = np.stack([r, theta], axis=-1)
    cart = conv.polar2cart(pol.copy(), degrees=degrees)
    pol_back = conv.cart2polar(cart.copy(), degrees=degrees)
    expected = pol.copy()
    assert_allclose(pol_back, expected, rtol=RTOL, atol=1e-9)
