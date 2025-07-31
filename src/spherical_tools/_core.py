import numpy as np
from numpy.typing import NDArray


def _cart2sph(cart: NDArray[np.float64]) -> NDArray[np.float64]:
    sph = np.empty_like(cart)  # initialise output array
    sph[..., 0] = np.linalg.norm(cart, axis=-1)  # radius
    sph[..., 1] = np.arctan2(cart[..., 1], cart[..., 0])  # azimuthal angle
    sph[..., 2] = np.arccos(cart[..., 2] / sph[..., 0])  # polar angle
    return sph


def _sph2cart(sph: NDArray[np.float64]) -> NDArray[np.float64]:
    cart = np.empty_like(sph)  # initialise output array
    cart[..., 0] = sph[..., 0] * np.cos(sph[..., 1]) * np.sin(sph[..., 2])  # x
    cart[..., 1] = sph[..., 0] * np.sin(sph[..., 1]) * np.sin(sph[..., 2])  # y
    cart[..., 2] = sph[..., 0] * np.cos(sph[..., 2])  # z
    return cart


def _sph2geo2(sph: NDArray[np.float64]) -> NDArray[np.float64]:
    geo = np.empty_like(sph)  # initialise output array
    geo[..., -2] = sph[..., -2]  # longitude/azimuthal angle remains the same
    geo[..., -1] = np.pi / 2 - sph[..., -1]  # convert colatitude to latitude
    return geo


def _sph2geo3(sph: NDArray[np.float64]) -> NDArray[np.float64]:
    geo = np.empty_like(sph)  # initialise output array
    geo[..., -3] = sph[..., -3]  # radius remains the same
    geo[..., -2] = sph[..., -2]  # longitude/azimuthal angle remains the same
    geo[..., -1] = np.pi / 2 - sph[..., -1]  # convert colatitude to latitude
    return geo


def _geo2sph2(geo: NDArray[np.float64]) -> NDArray[np.float64]:
    sph = np.empty_like(geo)  # initialise output array
    sph[..., -2] = geo[..., -2]  # longitude/azimuthal angle remains the same
    sph[..., -1] = np.pi / 2 - geo[..., -1]  # convert latitude to colatitude
    return sph


def _geo2sph3(geo: NDArray[np.float64]) -> NDArray[np.float64]:
    sph = np.empty_like(geo)  # initialise output array
    sph[..., -3] = geo[..., -3]  # radius remains the same
    sph[..., -2] = geo[..., -2]  # longitude/azimuthal angle remains the same
    sph[..., -1] = np.pi / 2 - geo[..., -1]  # convert latitude to colatitude
    return sph


def _cart2geo(cart: NDArray[np.float64]) -> NDArray[np.float64]:
    geo = np.empty_like(cart)  # initialise output array
    geo[..., 0] = np.linalg.norm(cart, axis=-1)  # radius
    geo[..., 1] = np.arctan2(cart[..., 1], cart[..., 0])  # longitude/azimuthal angle
    geo[..., 2] = np.arcsin(cart[..., 2] / geo[..., 0])  # latitude
    return geo


def _geo2cart(geo: NDArray[np.float64]) -> NDArray[np.float64]:
    cart = np.empty_like(geo)  # initialise output array
    cart[..., 0] = geo[..., 0] * np.cos(geo[..., 1]) * np.cos(geo[..., 2])  # x
    cart[..., 1] = geo[..., 0] * np.sin(geo[..., 1]) * np.cos(geo[..., 2])  # y
    cart[..., 2] = geo[..., 0] * np.sin(geo[..., 2])  # z
    return cart


def _cart2polar(cart: NDArray[np.float64]) -> NDArray[np.float64]:
    polar = np.empty_like(cart)  # initialise output array
    polar[..., 0] = np.linalg.norm(cart, axis=-1)  # radius
    polar[..., 1] = np.arctan2(cart[..., 1], cart[..., 0])  # azimuthal angle
    return polar


def _polar2cart(polar: NDArray[np.float64]) -> NDArray[np.float64]:
    cart = np.empty_like(polar)  # initialise output array
    cart[..., 0] = polar[..., 0] * np.cos(polar[..., 1])  # x
    cart[..., 1] = polar[..., 0] * np.sin(polar[..., 1])  # y
    return cart


def _unit_sphere_angle(
    arr1: NDArray[np.float64], arr2: NDArray[np.float64]
) -> NDArray[np.float64]:
    return 2.0 * np.arcsin(
        np.sqrt(
            (
                1
                - np.cos(arr2[..., 1] - arr1[..., 1])
                + np.sin(arr1[..., 1])
                * np.sin(arr2[..., 1])
                * (1 - np.cos(arr2[..., 0] - arr1[..., 0]))
            )
            / 2.0
        )
    )
