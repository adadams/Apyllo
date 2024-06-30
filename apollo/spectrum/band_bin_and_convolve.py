from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


def get_wavelengths_from_wavelength_bins(wavelength_bin_starts, wavelength_bin_ends):
    return (wavelength_bin_starts + wavelength_bin_ends) / 2


def get_wavelength_bins_from_wavelengths(wavelengths):
    half_wavelength_differences = np.diff(wavelengths) / 2

    wavelength_bin_starts = wavelengths - np.stack(
        (half_wavelength_differences[0], half_wavelength_differences)
    )
    wavelength_bin_ends = wavelengths + np.stack(
        (half_wavelength_differences, half_wavelength_differences[-1])
    )
    return wavelength_bin_starts, wavelength_bin_ends


def find_band_bounding_indices(
    wavelength_bin_starts: Sequence[float], wavelength_bin_ends: Sequence[float]
) -> tuple[int]:
    indices_where_bands_end: NDArray[np.int_] = np.argwhere(
        ~np.isin(wavelength_bin_ends, wavelength_bin_starts)
    ).squeeze()

    return (0, *(indices_where_bands_end + 1))


def find_band_limits_from_wavelength_bins(
    wavelength_bin_starts: Sequence[float], wavelength_bin_ends: Sequence[float]
) -> tuple[tuple[float, float]]:
    indices_bounding_bands: tuple[int] = find_band_bounding_indices(
        wavelength_bin_starts, wavelength_bin_ends
    )

    return (
        (wavelength_bin_starts[start], wavelength_bin_ends[end])
        for start, end in zip(indices_bounding_bands[:-1], indices_bounding_bands[1:])
    )


def find_band_slices_from_wavelength_bins(
    wavelength_bin_starts: Sequence[float], wavelength_bin_ends: Sequence[float]
) -> tuple[slice]:
    indices_bounding_bands: tuple[int] = find_band_bounding_indices(
        wavelength_bin_starts, wavelength_bin_ends
    )

    return tuple(
        slice(start, end)
        for start, end in zip(indices_bounding_bands[:-1], indices_bounding_bands[1:])
    )


@dataclass
class WavelengthBins:
    wavelength_bin_starts: NDArray[np.float_]
    wavelength_bin_ends: NDArray[np.float_]


@dataclass
class BinIndices:
    ibinlo: NDArray[np.int_]
    ibinhi: NDArray[np.int_]
    fbinlo: NDArray[np.float_]
    fbinhi: NDArray[np.float_]


def GetBinIndices(original_number_of_wavelengths: int, binw: int | float):
    blen = (int)(original_number_of_wavelengths / binw)
    ibinlo = np.zeros(blen)
    ibinhi = np.zeros(blen)
    fbinlo = np.zeros(blen)
    fbinhi = np.zeros(blen)

    for i in range(0, blen):
        ibinlo[i] = i * binw
        ibinhi[i] = (i + 1) * binw
        fbinlo[i] = np.modf(ibinlo[i])[0]
        fbinhi[i] = np.modf(ibinhi[i])[0]
        if fbinlo[i] == 0.0:
            ibinlo[i] = ibinlo[i] + 0.000_001
            fbinlo[i] = 0.000_001
        if fbinhi[i] == 0.0:
            ibinhi[i] = ibinhi[i] + 0.000_001
            fbinhi[i] = 0.000_001

    return BinIndices(ibinlo, ibinhi, fbinlo, fbinhi)


def BinWavelengths(
    wavelo: NDArray[np.float_],
    wavehi: NDArray[np.float_],
    bin_indices: BinIndices,
    binw: int | float,
) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
    if len(wavelo) != len(wavehi):
        raise ValueError(
            "Lower and upper wavelength bin arrays must be the same length"
        )

    blen = (int)(len(wavelo) / binw)
    print(f"{len(wavelo)=}")
    print(f"{len(bin_indices.ibinhi)=}")
    print(f"{binw=}")
    print(f"{blen=}")
    binlo = np.zeros(blen)
    binhi = np.zeros(blen)

    ibinlo = bin_indices.ibinlo
    ibinhi = bin_indices.ibinhi
    fbinlo = bin_indices.fbinlo
    fbinhi = bin_indices.fbinhi

    if len(wavelo) != len(wavehi):
        raise ValueError(
            "Lower and upper wavelength bin arrays must be the same length"
        )

    for i in range(0, len(ibinhi)):
        if (int)(np.floor(ibinlo[i])) == (int)(len(wavelo) - 1):
            binlo[i] = (1.0 - fbinlo[i]) * wavelo[(int)(np.floor(ibinlo[i]))]
        else:
            binlo[i] = (1.0 - fbinlo[i]) * wavelo[(int)(np.floor(ibinlo[i]))] + fbinlo[
                i
            ] * wavelo[(int)(np.floor(ibinlo[i])) + 1]

        if (int)(np.floor(ibinhi[i])) == len(wavelo):
            binhi[i] = (1.0 - fbinhi[i]) * wavehi[(int)(np.floor(ibinhi[i])) - 1]
        else:
            binhi[i] = (1.0 - fbinhi[i]) * wavehi[
                (int)(np.floor(ibinhi[i])) - 1
            ] + fbinhi[i] * wavehi[(int)(np.floor(ibinhi[i]))]

    return WavelengthBins(wavelength_bin_starts=binlo, wavelength_bin_ends=binhi)


def BinFlux(flux: ArrayLike, binw: int | float, bin_indices: BinIndices) -> ArrayLike:
    blen = (int)(len(flux) / binw)
    binflux = np.zeros(blen)

    ibinlo = bin_indices.ibinlo
    ibinhi = bin_indices.ibinhi
    fbinlo = bin_indices.fbinlo
    fbinhi = bin_indices.fbinhi

    for i in range(0, len(ibinhi)):
        binflux[i] = np.sum(
            flux[(int)(np.ceil(ibinlo[i])) : (int)(np.floor(ibinhi[i]))]
        )

        binflux[i] = binflux[i] + (1.0 - fbinlo[i]) * flux[(int)(np.floor(ibinlo[i]))]

        if (int)(np.floor(ibinhi[i])) >= len(flux):
            binflux[i] = binflux[i]
        else:
            binflux[i] = binflux[i] + fbinhi[i] * flux[(int)(np.floor(ibinhi[i]))]

        binflux[i] = binflux[i] / binw

    return binflux


def BinFluxErrors(
    flux_errors: ArrayLike, binw: int | float, bin_indices: BinIndices
) -> ArrayLike:
    binned_errors: ArrayLike = BinFlux(flux_errors, binw, bin_indices)
    print(f"{binned_errors=}")

    binned_errors_assuming_white_noise: ArrayLike = np.divide(
        binned_errors, np.sqrt(binw - 1)
    )
    return binned_errors_assuming_white_noise


def BinSpec(wavelo, wavehi, flux, binw):
    number_of_wavelengths: int = len(wavelo)
    bin_indices: BinIndices = GetBinIndices(number_of_wavelengths, binw)

    binned_wavelengths: WavelengthBins = BinWavelengths(
        wavelo, wavehi, bin_indices, binw
    )

    binflux: NDArray[np.float_] = BinFlux(flux, binw, bin_indices)

    return *binned_wavelengths, binflux


def BinSpecwithErrors(wavelo, wavehi, flux, err, binw):
    number_of_wavelengths: int = len(wavelo)
    bin_indices: BinIndices = GetBinIndices(number_of_wavelengths, binw)

    binned_wavelengths: WavelengthBins = BinWavelengths(
        wavelo, wavehi, bin_indices, binw
    )

    binflux: ArrayLike = BinFlux(flux, binw, bin_indices)
    binerr: ArrayLike = BinFluxErrors(err, binw, bin_indices)

    return *binned_wavelengths, binflux, binerr


def ConvSpec(flux: ArrayLike, bin_width: int | float) -> ArrayLike:
    # The factor of 6 is a somewhat arbitrary choice to get
    # a wide enough window for the Gaussian kernel
    kernel_width: float = bin_width * 6.0
    stdev: float = bin_width / 2.35

    kernel_remainder, kernel_integer = np.modf(kernel_width)
    if kernel_integer == 0:
        return flux

    kernel_range = np.arange(kernel_integer) + kernel_remainder - (kernel_width / 2)
    kernel = np.exp(-0.5 * (kernel_range / stdev) ** 2)
    kernel = kernel / np.sum(kernel)

    convflux = np.convolve(flux, kernel, mode="same")

    return convflux
