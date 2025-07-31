import numpy as np
from functools import wraps
from typing import Callable, Literal, Sequence, Union
from numpy.typing import ArrayLike, NDArray


def validate_coordinates(
    arr: ArrayLike, *, ndim: Union[int, Sequence[int]], name_in: str, name_out: str
) -> NDArray[np.float64]:
    """
    Convert arr to a float64 ndarray and check that last dimension matches ndim.

    Parameters
    ----------
    arr
        array-like input
    ndim
        allowed sizes for the last dimension
    name_in
        human-readable name for error messages
    name_out
        human-readable name for error messages

    Returns
    -------
    out
        array converted to float64 with validated shape
    """

    out = np.asarray(arr, dtype=np.float64)
    allowed = (ndim,) if isinstance(ndim, int) else tuple(ndim)
    if out.ndim < 1 or out.shape[-1] not in allowed:
        raise ValueError(
            f"Converting from {name_in!r} coordinates to {name_out!r} coordinates must have last dimension in {allowed}"
        )
    return out


def ensure_units(
    ndim: Union[int, Sequence[int]],
    name_in: str,
    name_out: str,
    *,
    convert_input: bool = False,
    convert_output: bool = False,
) -> Callable[[Callable[..., NDArray[np.float64]]], Callable[..., NDArray[np.float64]]]:
    """
    Decorator factory to validate coords and handle deg↔rad conversion.

    Parameters
    ----------
    ndim
        allowed sizes for the last dimension of the input array
    name_in
        human-readable name for error messages
    name_out
        human-readable name for error messages
    convert_input
        if True, on degrees=True convert last two entries from deg→rad
    convert_output
        if True, on degrees=True convert last two entries from rad→deg

    Returns
    -------
    decorator
        wraps a function that assumes all inputs/outputs in radians
    """

    def decorator(fn: Callable[..., NDArray[np.float64]]):
        @wraps(fn)
        def wrapper(
            arr: ArrayLike, *args, degrees: bool = False, **kwargs
        ) -> NDArray[np.float64]:
            # validate shape + dtype
            arr_rad = validate_coordinates(
                arr, ndim=ndim, name_in=name_in, name_out=name_out
            )
            # if requested, convert input angles from deg→rad
            if convert_input and degrees:
                arr_rad[..., -2:] = np.deg2rad(arr_rad[..., -2:])
            # core logic (always in radians)
            out = fn(arr_rad, *args, **kwargs)
            # if requested, convert output angles from rad→deg
            if convert_output and degrees:
                out[..., -2:] = np.rad2deg(out[..., -2:])
            return out

        return wrapper

    return decorator
