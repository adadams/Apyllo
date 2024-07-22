from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray


def get_wavelengths_from_wavelength_bins(wavelength_bin_starts, wavelength_bin_ends):
    return (wavelength_bin_starts + wavelength_bin_ends) / 2


def get_wavelength_bins_from_wavelengths(
    wavelengths: ArrayLike,
) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
    half_wavelength_differences = np.diff(wavelengths) / 2

    wavelength_bin_starts = wavelengths - np.hstack(
        (half_wavelength_differences[0], half_wavelength_differences)
    )
    wavelength_bin_ends = wavelengths + np.hstack(
        (half_wavelength_differences, half_wavelength_differences[-1])
    )
    return wavelength_bin_starts, wavelength_bin_ends


def find_band_bounding_indices(
    wavelength_bin_starts: Sequence[float], wavelength_bin_ends: Sequence[float]
) -> tuple[int]:
    indices_where_bands_end: NDArray[np.int_] = np.argwhere(
        ~np.isin(wavelength_bin_ends, wavelength_bin_starts)
    ).squeeze()

    if indices_where_bands_end.ndim == 0:
        # No bands found, the whole array is a single band
        return (0, indices_where_bands_end + 1)

    else:
        return (0, *(indices_where_bands_end + 1))


def find_band_limits_from_wavelength_bins(
    wavelength_bin_starts: Sequence[float], wavelength_bin_ends: Sequence[float]
) -> tuple[tuple[float, float]]:
    indices_bounding_bands: tuple[int] = find_band_bounding_indices(
        wavelength_bin_starts, wavelength_bin_ends
    )

    return tuple(
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


def GetBinIndices(original_number_of_wavelengths: int, binw: int | float) -> BinIndices:
    blen: int = int(original_number_of_wavelengths / binw)

    ibinlo: NDArray[np.float_] = np.arange(blen) * binw
    ibinhi: NDArray[np.float_] = (np.arange(blen) + 1) * binw

    fbinlo: NDArray[np.float_] = np.modf(ibinlo)[0]
    fbinhi: NDArray[np.float_] = np.modf(ibinhi)[0]

    SMALL_NUDGE: Final[float] = 1e-6
    ibinlo[fbinlo == 0.0] = ibinlo[fbinlo == 0.0] + SMALL_NUDGE
    ibinhi[fbinhi == 0.0] = ibinhi[fbinhi == 0.0] + SMALL_NUDGE
    fbinlo[fbinlo == 0.0] = SMALL_NUDGE
    fbinhi[fbinhi == 0.0] = SMALL_NUDGE

    return BinIndices(ibinlo, ibinhi, fbinlo, fbinhi)


def BinWavelengths(
    wavelo: ArrayLike, wavehi: ArrayLike, bin_indices: BinIndices
) -> WavelengthBins:
    if len(wavelo) != len(wavehi):
        raise ValueError(
            "Lower and upper wavelength bin arrays must be the same length"
        )

    lower_edge_indices: NDArray[np.int_] = np.floor(bin_indices.ibinlo).astype(int)
    fractional_lower_bin: NDArray[np.float_] = np.where(
        lower_edge_indices != len(wavelo) - 1, bin_indices.fbinlo, 0
    )

    binlo: NDArray[np.float_] = (1 - bin_indices.fbinlo) * np.take(
        wavelo, lower_edge_indices
    ) + fractional_lower_bin * np.take(wavelo, lower_edge_indices + 1)

    upper_edge_indices: NDArray[np.int_] = np.floor(bin_indices.ibinhi).astype(int)
    fractional_upper_bin: NDArray[np.float_] = np.where(
        upper_edge_indices != len(wavelo) - 1, bin_indices.fbinhi, 0
    )

    binhi: NDArray[np.float_] = (1 - bin_indices.fbinhi) * np.take(
        wavehi, upper_edge_indices - 1
    ) + fractional_upper_bin * np.take(wavehi, upper_edge_indices)

    return WavelengthBins(wavelength_bin_starts=binlo, wavelength_bin_ends=binhi)


def BinFlux(flux: ArrayLike, binw: int | float, bin_indices: BinIndices) -> ArrayLike:
    int_upper_ibinlo: NDArray[np.int_] = np.ceil(bin_indices.ibinlo).astype(int)
    int_lower_ibinhi: NDArray[np.int_] = np.floor(bin_indices.ibinhi).astype(int)

    lower_edge_indices: NDArray[np.int_] = np.floor(bin_indices.ibinlo).astype(int)
    upper_edge_indices: NDArray[np.int_] = int_lower_ibinhi

    binning_slices: NDArray[np.object_] = np.array(
        [slice(start, stop) for start, stop in zip(int_upper_ibinlo, int_lower_ibinhi)]
    )

    binflux_without_edges: NDArray[np.float_] = np.array(
        [np.sum(flux[slice]) for slice in binning_slices]
    )

    fractional_lower_bin: NDArray[np.float_] = 1 - bin_indices.fbinlo
    binflux_lower_edge: NDArray[np.float_] = fractional_lower_bin * np.take(
        flux, lower_edge_indices
    )

    fractional_upper_bin: NDArray[np.float_] = np.where(
        int_lower_ibinhi < len(flux), bin_indices.fbinhi, 0
    )
    binflux_upper_edge: NDArray[np.float_] = fractional_upper_bin * np.take(
        flux, upper_edge_indices
    )

    binflux = binflux_lower_edge + binflux_without_edges + binflux_upper_edge

    return binflux / binw


def BinFluxErrors(
    flux_errors: ArrayLike, binw: int | float, bin_indices: BinIndices
) -> ArrayLike:
    binned_errors: ArrayLike = BinFlux(flux_errors, binw, bin_indices)

    binned_errors_assuming_white_noise: ArrayLike = binned_errors / np.sqrt(binw - 1)
    return binned_errors_assuming_white_noise


def BinSpec(wavelo, wavehi, flux, binw):
    number_of_wavelengths: int = len(wavelo)
    bin_indices: BinIndices = GetBinIndices(number_of_wavelengths, binw)

    binned_wavelengths: WavelengthBins = BinWavelengths(
        wavelo, wavehi, bin_indices, binw
    )

    binflux: NDArray[np.float_] = BinFlux(flux, binw, bin_indices)

    return (
        binned_wavelengths.wavelength_bin_starts,
        binned_wavelengths.wavelength_bin_ends,
        binflux,
    )


def BinSpecwithErrors(wavelo, wavehi, flux, err, binw):
    number_of_wavelengths: int = len(wavelo)
    bin_indices: BinIndices = GetBinIndices(number_of_wavelengths, binw)

    binned_wavelengths: WavelengthBins = BinWavelengths(wavelo, wavehi, bin_indices)

    binflux: ArrayLike = BinFlux(flux, binw, bin_indices)
    binerr: ArrayLike = BinFluxErrors(err, binw, bin_indices)

    return (
        binned_wavelengths.wavelength_bin_starts,
        binned_wavelengths.wavelength_bin_ends,
        binflux,
        binerr,
    )


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
