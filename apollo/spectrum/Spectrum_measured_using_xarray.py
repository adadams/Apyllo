from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from xarray import Dataset

from apollo.convenience_types import Measured
from apollo.dataset.builders import make_dataset_variables_from_dict
from apollo.dataset.UsesXarray_ABC import UsesXarray
from apollo.spectrum.band_bin_and_convolve import (
    BinSpec,
    ConvSpec,
    get_wavelength_bins_from_wavelengths,
    get_wavelengths_from_wavelength_bins,
)
from apollo.spectrum.Spectrum_using_xarray import Spectrum


@dataclass
class DataSpectrum(Spectrum, Measured, UsesXarray):
    data: Dataset

    def __repr__(self) -> str:
        return (
            f"Spectrum for {self.data.attrs["title"]} {self.data} \n"
            + f"Pixel resolution: {self.mean_resolution.pint.magnitude:.2f} \n"
            + f"S/N per pixel: {self.mean_signal_to_noise.pint.magnitude:.2f}"
        )

    @property
    def mean_signal_to_noise(self) -> float:
        return np.mean(self.spectrum / self.errors)

    @classmethod
    def from_elements(
        cls,
        wavelength_bin_starts: NDArray[np.float_],
        wavelength_bin_ends: NDArray[np.float_],
        wavelengths: NDArray[np.float_],
        spectrum: NDArray[np.float_],
        lower_errors: NDArray[np.float_],
        upper_errors: NDArray[np.float_],
        wavelength_units="microns",
        spectral_units="erg/s/cm^3",
    ):
        data: Dataset = make_dataset_variables_from_dict(
            {
                "wavelength_bin_starts": wavelength_bin_starts,
                "wavelength_bin_ends": wavelength_bin_ends,
                "wavelength": wavelengths,
                "spectrum": spectrum,
                "lower_errors": lower_errors,
                "upper_errors": upper_errors,
            },
            dimension_names=["wavelength"],
            units={
                "wavelength": wavelength_units,
                "spectrum": spectral_units,
                "lower_errors": spectral_units,
                "upper_errors": spectral_units,
            },
        )
        return cls(data)

    @classmethod
    def from_wavelengths_only(
        cls,
        wavelengths: NDArray[np.float_],
        spectrum: NDArray[np.float_],
        lower_errors: NDArray[np.float_],
        upper_errors: NDArray[np.float_],
        wavelength_units: str,
        spectral_units: str,
    ):
        wavelength_bin_starts, wavelength_bin_ends = (
            get_wavelength_bins_from_wavelengths(wavelengths)
        )

        return cls.from_elements(
            wavelength_bin_starts,
            wavelength_bin_ends,
            wavelengths,
            spectrum,
            lower_errors,
            upper_errors,
            wavelength_units,
            spectral_units,
        )

    @classmethod
    def from_wavelength_bins_only(
        cls,
        wavelength_bin_starts: NDArray[np.float_],
        wavelength_bin_ends: NDArray[np.float_],
        spectrum: NDArray[np.float_],
        lower_errors: NDArray[np.float_],
        upper_errors: NDArray[np.float_],
        wavelength_units: str,
        spectral_units: str,
    ):
        wavelengths: NDArray[np.float_] = get_wavelengths_from_wavelength_bins(
            wavelength_bin_starts, wavelength_bin_ends
        )

        return cls.from_elements(
            wavelength_bin_starts,
            wavelength_bin_ends,
            wavelengths,
            spectrum,
            lower_errors,
            upper_errors,
            wavelength_units,
            spectral_units,
        )

    def convolve(
        self, convolve_factor: float | int, spectrum: ArrayLike = None
    ) -> list[NDArray[np.float_]]:
        if spectrum is None:
            spectrum = self.spectrum

        convolved_spectrum: NDArray[np.float_] = ConvSpec(
            self.spectrum, convolve_factor
        )

        convolved_errors: NDArray[np.float_] = ConvSpec(self.errors, convolve_factor)

        return convolved_spectrum, convolved_errors

    def down_bin(self, new_resolution: float | int, spectrum=None):
        if spectrum is None:
            spectrum = self.spectrum

        original_resolution: float = self.mean_resolution
        bin_factor: float = original_resolution / new_resolution

        # Bin the observations to fit a lower sampling resolution
        binned_spectrum, binned_errors, binned_WBS, binned_WBE = BinSpec(
            self.spectrum,
            self.errors,
            self.wavelength_bin_starts,
            self.wavelength_bin_ends,
            bin_factor,
        )

        return DataSpectrum.from_wavelength_bins_only(
            wavelength_bin_starts=binned_WBS,
            wavelength_bin_ends=binned_WBE,
            spectrum=binned_spectrum,
            lower_errors=binned_errors,
            upper_errors=binned_errors,
            wavelength_units=self.units_for("wavelengths"),
            spectral_units=self.units_for("spectrum"),
        )
