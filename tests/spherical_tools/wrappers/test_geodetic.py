"""
Tests for spherical_tools.wrappers.geodetic

These tests cover correctness, units handling, broadcasting, dateline handling,
and interpolation behavior for:
- great_circle_distance
- crosses_dateline
- fill_great_circle
"""

from __future__ import annotations

import numpy as np
import pytest

from spherical_tools.wrappers.geodetic import (
    great_circle_distance,
    crosses_dateline,
    fill_great_circle,
)


class TestGreatCircleDistance:
    def test_identical_points_zero_angle_radians(self) -> None:
        # same point -> zero central angle (radians)
        a = np.array([0.25 * np.pi, 0.5 * np.pi])  # spherical (θ, φ)
        ang = great_circle_distance(a, a, degrees=False, coordinate_system="spherical")
        assert ang == pytest.approx(0.0, abs=1e-15)

    def test_identical_points_zero_angle_degrees(self) -> None:
        # same point -> zero central angle (degrees)
        a = np.array([30.0, 45.0])  # geographic (lon, lat)
        ang = great_circle_distance(a, a, degrees=True, coordinate_system="geographic")
        assert ang == pytest.approx(0.0, abs=1e-12)

    def test_antipodes_angle_pi(self) -> None:
        # antipodal points -> 180° separation (π radians)
        a = np.array([0.0, 0.0])  # geographic (lon, lat)
        b = np.array([180.0, 0.0])  # antipode at equator
        ang = great_circle_distance(a, b, degrees=True, coordinate_system="geographic")
        assert ang == pytest.approx(180.0, rel=0, abs=1e-12)

    def test_with_radius_returns_arc_length(self) -> None:
        # with a radius, expect arc length in same units as radius
        r = 6371.0  # earth ~km (value itself not important)
        a = np.array([0.0, 0.0])
        b = np.array([90.0, 0.0])  # 90° separation
        d = great_circle_distance(
            a, b, degrees=True, coordinate_system="geographic", radius=r
        )
        assert d == pytest.approx(0.5 * np.pi * r, rel=1e-12)

    def test_degrees_affects_input_and_output_when_radius_none(self) -> None:
        # degrees=True should affect input parsing and angle output only
        a = np.array([0.0, 0.0])
        b = np.array([60.0, 0.0])
        ang_deg = great_circle_distance(
            a, b, degrees=True, coordinate_system="geographic", radius=None
        )
        ang_rad = great_circle_distance(
            np.deg2rad(a), np.deg2rad(b), degrees=False, coordinate_system="geographic"
        )
        assert ang_deg == pytest.approx(np.rad2deg(ang_rad), rel=1e-12)

    def test_broadcasting_last_axis_two(self) -> None:
        # broadcasting across leading dims should work
        arr1 = np.array([[0.0, 0.0], [0.0, 0.0]])
        arr2 = np.array([90.0, 0.0])
        out = great_circle_distance(
            arr1, arr2, degrees=True, coordinate_system="geographic"
        )
        assert out.shape == (2,)
        assert np.allclose(out, 90.0, atol=1e-12)

    def test_invalid_coordinate_system_raises(self) -> None:
        a = np.array([0.0, 0.0])
        b = np.array([10.0, 0.0])
        with pytest.raises(ValueError):
            great_circle_distance(a, b, coordinate_system="cartesian")


class TestCrossesDateline:
    def test_simple_cross_true(self) -> None:
        # 170E -> 170W should cross
        a = np.array([170.0, 0.0])
        b = np.array([-170.0, 0.0])
        out = crosses_dateline(a, b, degrees=True, coordinate_system="geographic")
        assert bool(out) is True

    def test_simple_cross_false(self) -> None:
        a = np.array([10.0, 0.0])
        b = np.array([30.0, 0.0])
        out = crosses_dateline(a, b, degrees=True, coordinate_system="geographic")
        assert bool(out) is False

    def test_near_wrap_cross_true(self) -> None:
        a = np.array([179.0, 5.0])
        b = np.array([-179.0, -5.0])
        out = crosses_dateline(a, b, degrees=True, coordinate_system="geographic")
        assert bool(out) is True

    def test_spherical_theta_used(self) -> None:
        # use spherical (θ, φ) with θ analogous to longitude
        a = np.array([np.deg2rad(170.0), np.deg2rad(90.0)])  # equator
        b = np.array([np.deg2rad(-170.0), np.deg2rad(90.0)])
        out = crosses_dateline(a, b, degrees=False, coordinate_system="spherical")
        assert bool(out) is True

    def test_invalid_coordinate_system_raises(self) -> None:
        a = np.array([0.0, 0.0])
        b = np.array([0.0, 0.0])
        with pytest.raises(ValueError):
            crosses_dateline(a, b, coordinate_system="cartesian")


class TestFillGreatCircle:
    def test_equator_quarter_arc_defaults(self) -> None:
        # default args are geographic + degrees
        a = np.array([0.0, 0.0])  # lon, lat
        b = np.array([90.0, 0.0])
        prof = fill_great_circle(a, b, res=10.0)  # n_points computed from res
        # 90/10 = 9 segments => 10 points
        assert prof.shape == (10, 2)
        # lat should be ~0 everywhere; lon monotonic 0..90
        assert np.allclose(prof[:, 1], 0.0, atol=1e-12)
        assert prof[0, 0] == pytest.approx(0.0, abs=1e-12)
        assert prof[-1, 0] == pytest.approx(90.0, abs=1e-12)
        assert np.all(np.diff(prof[:, 0]) > 0.0)

    def test_n_points_overrides_res_and_includes_endpoints(self) -> None:
        a = np.array([0.0, 0.0])
        b = np.array([90.0, 0.0])
        prof = fill_great_circle(a, b, res=0.1, n_points=5)  # res ignored
        assert prof.shape == (5, 2)
        # endpoints preserved
        assert prof[0, 0] == pytest.approx(0.0, abs=1e-12)
        assert prof[-1, 0] == pytest.approx(90.0, abs=1e-12)

    def test_return_angle_matches_gcd(self) -> None:
        a = np.array([20.0, 10.0])
        b = np.array([85.0, 12.0])
        prof, ang = fill_great_circle(a, b, res=5.0, return_angle=True)
        ang2 = great_circle_distance(a, b, degrees=True, coordinate_system="geographic")
        assert ang == pytest.approx(ang2, rel=1e-12)

    def test_dateline_unwrap_monotonic(self) -> None:
        # path 170E -> 170W should unwrap to 170..190
        a = np.array([170.0, 0.0])
        b = np.array([-170.0, 0.0])
        prof = fill_great_circle(a, b, res=2.0)
        lons = prof[:, 0]
        assert lons[0] == pytest.approx(170.0, abs=1e-10)
        assert lons[-1] == pytest.approx(190.0, abs=1e-10)
        assert np.all(np.diff(lons) > 0.0)

    def test_spherical_inputs_keep_phi_constant_on_equator(self) -> None:
        # spherical (θ, φ) with φ=π/2 (equator); θ: 0 -> π/2
        a = np.array([0.0, 0.5 * np.pi])
        b = np.array([0.5 * np.pi, 0.5 * np.pi])
        prof = fill_great_circle(
            a, b, res=np.deg2rad(10.0), degrees=False, coordinate_system="spherical"
        )
        # φ should stay ~π/2
        assert np.allclose(prof[:, 1], 0.5 * np.pi, atol=1e-12)
        # θ goes 0..π/2
        assert prof[0, 0] == pytest.approx(0.0, abs=1e-12)
        assert prof[-1, 0] == pytest.approx(0.5 * np.pi, abs=1e-12)

    def test_identical_points_returns_two_points_by_default(self) -> None:
        a = np.array([42.0, -10.0])
        prof = fill_great_circle(a, a, res=10.0)
        assert prof.shape == (2, 2)
        assert np.allclose(prof[0], a)
        assert np.allclose(prof[1], a)

    def test_invalid_coordinate_system_raises(self) -> None:
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])
        with pytest.raises(ValueError):
            fill_great_circle(a, b, coordinate_system="cartesian")

    @pytest.mark.parametrize("bad_n", [0, 1])
    def test_bad_n_points_raises(self, bad_n: int) -> None:
        a = np.array([0.0, 0.0])
        b = np.array([10.0, 0.0])
        with pytest.raises(ValueError):
            fill_great_circle(a, b, res=1.0, n_points=bad_n)

    def test_nonfloat_res_raises(self) -> None:
        a = np.array([0.0, 0.0])
        b = np.array([10.0, 0.0])
        with pytest.raises(ValueError):
            fill_great_circle(a, b, res="1.0")  # type: ignore[arg-type]

    def test_nonfloat_tol_raises(self) -> None:
        a = np.array([0.0, 0.0])
        b = np.array([10.0, 0.0])
        with pytest.raises(ValueError):
            fill_great_circle(a, b, res=1.0, tol=1)  # type: ignore[arg-type]
