import sys
from abc import abstractmethod
from dataclasses import dataclass
from os.path import abspath
from typing import Any, Protocol

import astropy.units as u
import numpy as np
from numpy.typing import NDArray
from xarray import Dataset

APOLLO_DIRECTORY = abspath(
    "/Users/arthur/Documents/Astronomy/2019/Retrieval/Code/APOLLO"
)
if APOLLO_DIRECTORY not in sys.path:
    sys.path.append(APOLLO_DIRECTORY)

from apollo.convenience_types import Measured  # noqa: E402
from apollo.dataset.dataset_functions import (  # noqa: E402
    make_dataset_variables_from_dict,  # noqa: E402
)

MICRONS = u.um
APOLLO_FLUX_UNITS = u.erg / u.s / u.cm**3

SPECTRAL_UNITS = dict(
    wavelength_bin_starts=MICRONS,
    wavelength_bin_ends=MICRONS,
    wavelengths=MICRONS,
    spectrum=APOLLO_FLUX_UNITS,
    lower_errors=APOLLO_FLUX_UNITS,
    upper_errors=APOLLO_FLUX_UNITS,
    spectrum_with_noise=APOLLO_FLUX_UNITS,
)


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


@dataclass
class Spectral(Protocol):
    """Has wavelengths, possibly in binned, possibly in unbinned input format.
    Also has a corresponding spectrum that can be convolved and down-sampled in
    resolution."""

    wavelength_bin_starts: NDArray[np.float64]
    wavelength_bin_ends: NDArray[np.float64]
    wavelengths: NDArray[np.float64]
    spectrum: NDArray[np.float64]

    @property
    def resolution(self, pixels_per_resolution_element=1):
        return self.wavelengths / (
            self.wavelength_bin_ends - self.wavelength_bin_starts
        )

    @abstractmethod
    def convolve(self, *args, **kwargs) -> NDArray[np.float64]:
        raise NotImplementedError

    @abstractmethod
    def down_bin(self, *args, **kwargs) -> NDArray[np.float64]:
        raise NotImplementedError

    @abstractmethod
    def down_resolve(self, *args, **kwargs) -> NDArray[np.float64]:
        raise NotImplementedError


class Spectrum(Spectral):
    def __init__(self, dataset: Dataset):
        self._dataset: Dataset = dataset

    def __getattr__(self, __name: str) -> Any:
        return self._dataset.get(__name)

    def __repr__(self) -> str:
        return (
            f"pixel resolution: {self.mean_resolution:.2f}, " + f"data: {self._dataset}"
        )

    @property
    def get_wavelength_bin_starts(self) -> NDArray[np.float64]:
        return self.wavelength_bin_starts

    @property
    def get_wavelength_bin_ends(self) -> NDArray[np.float64]:
        return self.wavelength_bin_ends

    @property
    def get_wavelengths(self) -> NDArray[np.float64]:
        return self.wavelengths

    @property
    def get_spectrum(self) -> NDArray[np.float64]:
        return self.spectrum

    @classmethod
    def from_wavelengths(
        cls,
        *,
        wavelengths: NDArray[np.float64],
        wavelength_units: str = "microns",
        spectrum: NDArray[np.float64],
        spectral_units: str = "erg/s/cm^3",
    ):
        wavelength_bin_starts, wavelength_bin_ends = (
            get_wavelength_bins_from_wavelengths(wavelengths)
        )

        # Note: correct the syntax.
        dataset: Dataset = make_dataset_variables_from_dict(
            dict(
                wavelength_bin_starts=wavelength_bin_starts,
                wavelength_bin_ends=wavelength_bin_ends,
                wavelength=wavelengths,
                spectrum=spectrum,
            ),
            dimension_names=["wavelength"],
            units=dict(wavelength=wavelength_units, spectrum=spectral_units),
        )

        return cls(dataset)

    @classmethod
    def from_wavelength_bins(
        cls,
        *,
        wavelength_bin_starts: NDArray[np.float64],
        wavelength_bin_ends: NDArray[np.float64],
        wavelength_units: str = "microns",
        spectrum: NDArray[np.float64],
        spectral_units: str = "erg/s/cm^3",
    ):
        wavelengths = get_wavelengths_from_wavelength_bins(
            wavelength_bin_starts, wavelength_bin_ends
        )

        # Note: correct the syntax.
        dataset: Dataset = make_dataset_variables_from_dict(
            dict(
                wavelength_bin_starts=wavelength_bin_starts,
                wavelength_bin_ends=wavelength_bin_ends,
                wavelength=wavelengths,
                spectrum=spectrum,
            ),
            dimension_names=["wavelength"],
            units=dict(wavelength=wavelength_units, spectrum=spectral_units),
        )

        return cls(dataset)

    @property
    def mean_resolution(self):
        return self.resolution.mean()

    def convolve(self, convolve_factor):
        convolved_spectrum = ConvSpec(self.spectrum, convolve_factor)

        return self.wavelength_bin_starts, self.wavelength_bin_ends, convolved_spectrum

    def down_bin(
        self, wavelength_bin_starts, wavelength_bin_ends, spectrum, new_resolution
    ):
        original_resolution = self.mean_resolution
        bin_factor = original_resolution / new_resolution

        # Bin the observations to fit a lower sampling resolution
        binned_spectrum, _there_are_no_errors_, binned_WBS, binned_WBE = BinSpec(
            spectrum,
            np.zeros_like(spectrum),
            wavelength_bin_starts,
            wavelength_bin_ends,
            bin_factor,
        )

        return Spectrum.from_binned_data(
            wavelength_bin_starts=binned_WBS,
            wavelength_bin_ends=binned_WBE,
            spectrum=binned_spectrum,
        )

    def down_resolve(self, convolve_factor, new_resolution):
        return self.down_bin(*self.convolve(convolve_factor), new_resolution)


@dataclass
class DataSpectrum(Spectrum, Measured):
    def __repr__(self) -> str:
        unit_prints = {
            value_name: unit.to_string()
            for value_name, unit in SPECTRAL_UNITS.items()
            if value_name in self.dataframe.columns
        }
        return (
            f"pixel resolution: {self.mean_resolution:.2f}, "
            + f"S/N per pixel: {self.mean_signal_to_noise:.2f}, "
            + f"data: {unit_prints} {self.dataframe}"
        )

    @property
    def errors(self):
        return (self.lower_errors + self.upper_errors) / 2

    @property
    def mean_signal_to_noise(self):
        return np.asarray(self.spectrum / self.errors).mean()

    def convolve(self, convolve_factor):
        convolved_spectrum = ConvSpec(self.spectrum, convolve_factor)
        convolved_errors = ConvSpec(self.errors, convolve_factor)

        return (
            self.wavelength_bin_starts,
            self.wavelength_bin_ends,
            convolved_spectrum,
            convolved_errors,
        )

    def down_bin(
        self,
        wavelength_bin_starts,
        wavelength_bin_ends,
        spectrum,
        errors,
        new_resolution,
    ):
        original_resolution = self.mean_resolution
        bin_factor = original_resolution / new_resolution

        # Bin the observations to fit a lower sampling resolution
        binned_spectrum, binned_errors, binned_WBS, binned_WBE = BinSpec(
            spectrum, errors, wavelength_bin_starts, wavelength_bin_ends, bin_factor
        )
        binned_wavelengths = get_wavelengths_from_wavelength_bins(
            binned_WBS, binned_WBE
        )

        return DataSpectrum.from_elements(
            wavelength_bin_starts=binned_WBS,
            wavelength_bin_ends=binned_WBE,
            wavelengths=binned_wavelengths,
            spectrum=binned_spectrum,
            lower_errors=binned_errors,
            upper_errors=binned_errors,
        )


def BinSpec(flux, err, wavelo, wavehi, binw):
    blen = (int)(len(flux) / binw)
    binflux = np.zeros(blen)
    binerr = np.zeros(blen)
    ibinlo = np.zeros(blen)
    ibinhi = np.zeros(blen)
    fbinlo = np.zeros(blen)
    fbinhi = np.zeros(blen)
    binlo = np.zeros(blen)
    binhi = np.zeros(blen)

    for i in range(0, blen):
        ibinlo[i] = i * binw
        ibinhi[i] = (i + 1) * binw
        fbinlo[i] = np.modf(ibinlo[i])[0]
        fbinhi[i] = np.modf(ibinhi[i])[0]
        if fbinlo[i] == 0.0:
            ibinlo[i] = ibinlo[i] + 0.000001
            fbinlo[i] = 0.000001
        if fbinhi[i] == 0.0:
            ibinhi[i] = ibinhi[i] + 0.000001
            fbinhi[i] = 0.000001

    for i in range(0, len(ibinhi)):
        binflux[i] = np.sum(
            flux[(int)(np.ceil(ibinlo[i])) : (int)(np.floor(ibinhi[i]))]
        )
        binerr[i] = np.sum(err[(int)(np.ceil(ibinlo[i])) : (int)(np.floor(ibinhi[i]))])

        binflux[i] = binflux[i] + (1.0 - fbinlo[i]) * flux[(int)(np.floor(ibinlo[i]))]
        binerr[i] = binerr[i] + (1.0 - fbinlo[i]) * err[(int)(np.floor(ibinlo[i]))]

        if (int)(np.floor(ibinlo[i])) == (int)(len(flux) - 1):
            binlo[i] = (1.0 - fbinlo[i]) * wavelo[(int)(np.floor(ibinlo[i]))]
        else:
            binlo[i] = (1.0 - fbinlo[i]) * wavelo[(int)(np.floor(ibinlo[i]))] + fbinlo[
                i
            ] * wavelo[(int)(np.floor(ibinlo[i])) + 1]

        if (int)(np.floor(ibinhi[i])) == len(flux):
            binhi[i] = (1.0 - fbinhi[i]) * wavehi[(int)(np.floor(ibinhi[i])) - 1]
        else:
            binhi[i] = (1.0 - fbinhi[i]) * wavehi[
                (int)(np.floor(ibinhi[i])) - 1
            ] + fbinhi[i] * wavehi[(int)(np.floor(ibinhi[i]))]

        if (int)(np.floor(ibinhi[i])) >= len(flux):
            binflux[i] = binflux[i]
            binerr[i] = binerr[i]
        else:
            binflux[i] = binflux[i] + fbinhi[i] * flux[(int)(np.floor(ibinhi[i]))]
            binerr[i] = binerr[i] + fbinhi[i] * err[(int)(np.floor(ibinhi[i]))]

        binflux[i] = binflux[i] / binw
        binerr[i] = binerr[i] / binw
        binerr[i] = binerr[i] / np.sqrt(binw - 1.0)

    return binflux, binerr, binlo, binhi


def ConvSpec(flux, bin_width):
    # The factor of 6 is a somewhat arbitrary choice to get
    # a wide enough window for the Gaussian kernel
    kernel_width = bin_width * 6.0
    stdev = bin_width / 2.35

    kernel_remainder, kernel_integer = np.modf(kernel_width)
    if kernel_integer == 0:
        return flux

    kernel_range = np.arange(kernel_integer) + kernel_remainder - (kernel_width / 2)
    kernel = np.exp(-0.5 * (kernel_range / stdev) ** 2)
    kernel = kernel / np.sum(kernel)

    convflux = np.convolve(flux, kernel, mode="same")

    return convflux
