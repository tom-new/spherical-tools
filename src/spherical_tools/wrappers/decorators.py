# decorators.py
from __future__ import annotations
from functools import wraps
from typing import Callable, Sequence, Union
import numpy as np
from numpy.typing import ArrayLike, NDArray

AngleSpec = Union[
    None, int, slice, Sequence[int], dict[int, Union[int, slice, Sequence[int]]]
]


def _angle_idx(spec: AngleSpec, lastdim: int) -> AngleSpec:
    """Resolve an angle index spec for a given last dimension."""
    if spec is None:
        return None
    if isinstance(spec, dict):
        return spec.get(lastdim, None)
    return spec


def validate_coordinates(
    arr: ArrayLike, *, ndim: Union[int, Sequence[int]], name_in: str, name_out: str
) -> NDArray[np.float64]:
    """
    Convert arr to float64 ndarray and check last dimension size(s).

    Parameters
    ----------
    arr
        array-like input.
    ndim
        allowed sizes for the last dimension.
    name_in, name_out
        names used in error messages.

    Returns
    -------
    out : ndarray
        view/copy of input as float64 ndarray.

    Raises
    ------
    ValueError
        if last dimension is not in ``ndim``.
    """
    out = np.asarray(arr, dtype=np.float64)
    allowed = (ndim,) if isinstance(ndim, int) else tuple(ndim)
    if out.ndim == 0 or out.shape[-1] not in allowed:
        raise ValueError(
            f"Converting from '{name_in}' coordinates to '{name_out}' coordinates "
            f"must have last dimension in {allowed}"
        )
    return out


def ensure_units(
    ndim: Union[int, Sequence[int]],
    name_in: str,
    name_out: str,
    *,
    convert_input: bool = False,
    convert_output: bool = False,
    angles_in: AngleSpec = None,
    angles_out: AngleSpec = None,
) -> Callable[[Callable[..., NDArray[np.float64]]], Callable[..., NDArray[np.float64]]]:
    """
    Decorator to validate coords and handle deg↔rad conversion using angle indices.

    Parameters
    ----------
    ndim
        allowed last-dimension sizes for the input.
    name_in, name_out
        coordinate system names for error messages.
    convert_input, convert_output
        whether to convert input and/or output angles when ``degrees=True``.
    angles_in, angles_out
        which indices are angles for input and output, respectively.
        May be:
          * a single int, slice, or sequence of ints;
          * a dict mapping last-dimension size to one of the above;
          * or ``None`` (no angles).

        Examples
        --------
        * spherical 3-vector (r, θ, φ): ``{3: (1, 2)}``
        * spherical 2-vector (θ, φ): ``{2: (0, 1)}``
        * polar 2-vector (r, θ): ``{2: (1,)}``

    Returns
    -------
    wrapper : Callable
        function that enforces validation and optional unit conversion.
    """

    def decorator(fn: Callable[..., NDArray[np.float64]]):
        @wraps(fn)
        def wrapper(
            arr: ArrayLike, *args, degrees: bool = False, **kwargs
        ) -> NDArray[np.float64]:
            # always validate shape and make a safe working copy
            arr64 = validate_coordinates(
                arr, ndim=ndim, name_in=name_in, name_out=name_out
            )
            arr_rad = arr64.copy()  # don't mutate caller's array
            # optional input conversion
            if convert_input and degrees:
                idx = _angle_idx(angles_in, arr_rad.shape[-1])
                if idx is not None:
                    arr_rad[..., idx] = np.deg2rad(arr_rad[..., idx])
            # core logic (expects radians)
            out = fn(arr_rad, *args, **kwargs)
            # optional output conversion
            if convert_output and degrees:
                idx = _angle_idx(angles_out, out.shape[-1])
                if idx is not None:
                    out = np.array(out, copy=True)  # avoid mutating callee's buffer
                    out[..., idx] = np.rad2deg(out[..., idx])
            return out

        return wrapper

    return decorator
