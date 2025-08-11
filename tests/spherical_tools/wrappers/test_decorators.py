import numpy as np
import pytest
from numpy.testing import assert_allclose

from spherical_tools.wrappers.decorators import validate_coordinates, ensure_units

class TestValidateCoordinates:

    def test_validate_coordinates_accepts_list_and_casts_float64(self):
        # list input should be accepted and returned as float64 ndarray
        out = validate_coordinates(
            [1, 0, 0], ndim=3, name_in="Cartesian", name_out="spherical"
        )
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.float64
        assert_allclose(out, np.array([1.0, 0.0, 0.0]))

    def test_validate_coordinates_rejects_wrong_last_dim(self):
        with pytest.raises(ValueError, match="last dimension"):
            validate_coordinates(np.ones(4), ndim=(2, 3), name_in="foo", name_out="bar")

    def test_valid_int_ndim_keeps_dtype_and_shape(self):
        # when ndim is an int, last dimension must equal it
        arr = [[1, 2, 3], [4, 5, 6]]
        out = validate_coordinates(arr, ndim=3, name_in="in", name_out="out")
        # dtype should be float64
        assert out.dtype == np.float64
        # shape should match input
        assert out.shape == (2, 3)
        # contents should be equal after conversion
        np.testing.assert_allclose(out, np.asarray(arr, dtype=np.float64))

    def test_valid_sequence_ndim_allows_any_of_them(self):
        # when ndim is a sequence, last dim may be any in that sequence
        # here allow both 2 and 3
        arr2 = [[1, 2], [3, 4]]
        out2 = validate_coordinates(arr2, ndim=(2, 3), name_in="in", name_out="out")
        assert out2.shape[-1] == 2

        arr3 = np.array([1.0, 2.0, 3.0])
        out3 = validate_coordinates(arr3, ndim=(2, 3), name_in="in", name_out="out")
        # 1D input of length 3 is allowed
        assert out3.ndim == 1 and out3.shape[-1] == 3

    @pytest.mark.parametrize(
        "bad_arr, ndim",
        [
            ([], 1),  # zero-length -> last dim 0 not allowed
            ([[1, 2, 3, 4]], 3),  # last dim = 4 ≠ 3
            ([[1], [2]], (2, 3)),  # last dim = 1 not in (2, 3)
        ],
    )
    def test_invalid_shape_raises_value_error(self, bad_arr, ndim):
        with pytest.raises(ValueError) as exc:
            validate_coordinates(bad_arr, ndim=ndim, name_in="foo", name_out="bar")
        msg = str(exc.value)
        # error message mentions both names and the allowed dims
        assert "foo" in msg and "bar" in msg
        allowed = (ndim,) if isinstance(ndim, int) else tuple(ndim)
        assert str(allowed) in msg

    def test_inputs_with_more_dims_pass_if_last_dim_matches(self):
        # 3D array whose last dim is 3 is fine
        arr3d = np.ones((4, 5, 3))
        out = validate_coordinates(arr3d, ndim=3, name_in="a", name_out="b")
        assert out.shape == (4, 5, 3)
        # 4D array whose last dim is 2 is fine when ndim=(2,3)
        arr4d = np.zeros((2, 2, 2, 2))
        out4d = validate_coordinates(arr4d, ndim=(2, 3), name_in="a", name_out="b")
        assert out4d.shape[-1] == 2


class TestEnsureUnits:

    def test_does_not_mutate_input_when_converting_degrees(self):
        def core(x):  # identity core
            return x

        wrapped = ensure_units(
            ndim=3,
            name_in="spherical",
            name_out="spherical",
            convert_input=True,
            convert_output=True,
            angles_in={3: (1, 2)},
            angles_out={3: (1, 2)},
        )(core)

        arr = np.array([1.0, 90.0, 0.0])  # degrees
        before = arr.copy()
        _ = wrapped(arr, degrees=True)
        assert_allclose(arr, before)

    def test_no_conversion_degrees_false(self):
        @ensure_units(
            ndim=3,
            name_in="in",
            name_out="out",
            convert_input=True,
            convert_output=True,
            angles_in={3: (1, 2)},
            angles_out={3: (1, 2)},
        )
        def core(arr):
            return arr.copy()

        arr = [1, 2, 3]
        out = core(arr, degrees=False)
        np.testing.assert_allclose(out, np.asarray(arr, dtype=np.float64))

    def test_no_conversion_flags_false_but_degrees_true(self):
        @ensure_units(
            ndim=3, name_in="in", name_out="out", convert_input=False, convert_output=False
        )
        def core(arr):
            return arr.copy()

        arr = [1, 2, 3]
        out = core(arr, degrees=True)
        np.testing.assert_allclose(out, np.asarray(arr, dtype=np.float64))

    def test_convert_input_only(self):
        @ensure_units(
            ndim=3,
            name_in="in",
            name_out="out",
            convert_input=True,
            convert_output=False,
            angles_in={3: (1, 2)},
        )
        def core(arr):
            return arr.copy()

        arr_deg = np.array([1.0, 90.0, 45.0])
        out = core(arr_deg, degrees=True)
        # last two entries converted to radians
        expected = np.array([1.0, np.deg2rad(90.0), np.deg2rad(45.0)])
        np.testing.assert_allclose(out, expected)

        # degrees=False should bypass conversion
        out2 = core(arr_deg, degrees=False)
        np.testing.assert_allclose(out2, np.asarray(arr_deg, dtype=np.float64))

    def test_convert_output_only(self):
        @ensure_units(
            ndim=3,
            name_in="in",
            name_out="out",
            convert_input=False,
            convert_output=True,
            angles_out={3: (1, 2)},
        )
        def core(arr):
            return arr.copy()

        arr_rad = np.array([1.0, np.pi / 2, np.pi / 4])
        out = core(arr_rad, degrees=True)
        # last two entries converted to degrees
        expected = np.array([1.0, 90.0, 45.0])
        np.testing.assert_allclose(out, expected)

        # degrees=False: no conversion
        out2 = core(arr_rad, degrees=False)
        np.testing.assert_allclose(out2, arr_rad)

    def test_both_conversions_with_additional_args(self):
        @ensure_units(
            ndim=3,
            name_in="in",
            name_out="out",
            convert_input=True,
            convert_output=True,
            angles_in={3: (1, 2)},
            angles_out={3: (1, 2)},
        )
        def core(arr, factor):
            arr2 = arr.copy()
            arr2[0] *= factor
            return arr2

        arr_deg = np.array([2.0, 90.0, 45.0])
        out = core(arr_deg, 3.0, degrees=True)
        # input arr_deg: [2.0, 90.0, 45.0]
        # after input conversion: [2.0, pi/2, pi/4]
        # multiply radius: [6.0, pi/2, pi/4]
        # output conversion: last two back to degrees: [6.0, 90.0, 45.0]
        expected = np.array([6.0, 90.0, 45.0])
        np.testing.assert_allclose(out, expected)

    def test_2d_array_convert_input(self):
        @ensure_units(
            ndim=3, name_in="in", name_out="out", convert_input=True, angles_in={3: (1, 2)}
        )
        def core(arr):
            return arr

        arr_deg_2d = np.array([[1, 90, 45], [2, 0, 180]])
        out = core(arr_deg_2d, degrees=True)
        expected = np.array(
            [
                [1, np.deg2rad(90), np.deg2rad(45)],
                [2, np.deg2rad(0), np.deg2rad(180)],
            ]
        )
        np.testing.assert_allclose(out, expected)

    def test_invalid_shape_raises_value_error(self):
        @ensure_units(ndim=3, name_in="a", name_out="b")
        def core(arr):
            return arr

        with pytest.raises(ValueError) as exc:
            core([1, 2], degrees=True)
        msg = str(exc.value)
        assert "a" in msg and "b" in msg
