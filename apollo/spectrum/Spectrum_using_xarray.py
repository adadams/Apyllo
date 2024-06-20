from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import NDArray
from xarray import DataArray, Dataset

from apollo.dataset.builders import make_dataset_variables_from_dict
from apollo.dataset.UsesXarray_ABC import UsesXarray
from apollo.spectrum.band_bin_and_convolve import (
    BinSpec,
    ConvSpec,
    get_wavelength_bins_from_wavelengths,
    get_wavelengths_from_wavelength_bins,
)
from apollo.spectrum.Spectral_protocol import Spectral


@dataclass
class Spectrum(Spectral, UsesXarray):
    def __getattr__(self, __name: str) -> DataArray:
        return self.data.get(__name)

    def __repr__(self) -> str:
        return (
            f"Spectrum for {self.data.attrs["title"]} {self.data} \n"
            + f"Pixel resolution: {self.mean_resolution.pint.magnitude:.2f}"
        )

    @property
    def pixel_resolution(self) -> float:
        return self.wavelengths / (
            self.wavelength_bin_ends - self.wavelength_bin_starts
        )

    @property
    def mean_resolution(self, pixels_per_resolution_element=1) -> float:
        return np.mean(self.pixel_resolution / pixels_per_resolution_element)

    @classmethod
    def from_elements(
        cls,
        wavelength_bin_starts: NDArray[np.float_],
        wavelength_bin_ends: NDArray[np.float_],
        wavelengths: NDArray[np.float_],
        spectrum: NDArray[np.float_],
        wavelength_units="microns",
        spectral_units="erg/s/cm^3",
    ):
        data: Dataset = make_dataset_variables_from_dict(
            {
                "wavelength_bin_starts": wavelength_bin_starts,
                "wavelength_bin_ends": wavelength_bin_ends,
                "wavelength": wavelengths,
                "spectrum": spectrum,
            },
            dimension_names=["wavelength"],
            units={"wavelength": wavelength_units, "spectrum": spectral_units},
        )
        return cls(data)

    @classmethod
    def from_wavelengths_only(
        cls,
        wavelengths: NDArray[np.float_],
        spectrum: NDArray[np.float_],
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
            wavelength_units,
            spectral_units,
        )

    @classmethod
    def from_wavelength_bins_only(
        cls,
        wavelength_bin_starts: NDArray[np.float_],
        wavelength_bin_ends: NDArray[np.float_],
        spectrum: NDArray[np.float_],
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
            wavelength_units,
            spectral_units,
        )

    def units_for(self, variable_name: str) -> str:
        return self.data[variable_name].attrs["units"]

    def convolve(self, convolve_factor: float | int) -> NDArray[np.float_]:
        return ConvSpec(self.spectrum, convolve_factor)

    def down_bin(self, new_resolution, spectrum=None) -> Self:
        if spectrum is None:
            spectrum = self.spectrum

        original_resolution: float = self.mean_resolution
        bin_factor: float | int = original_resolution / new_resolution

        # Bin the observations to fit a lower sampling resolution
        binned_spectrum, _there_are_no_errors_, binned_WBS, binned_WBE = BinSpec(
            spectrum,
            np.zeros_like(spectrum),
            self.wavelength_bin_starts,
            self.wavelength_bin_ends,
            bin_factor,
        )

        return Spectrum.from_wavelength_bins_only(
            wavelength_bin_starts=binned_WBS,
            wavelength_bin_ends=binned_WBE,
            spectrum=binned_spectrum,
            wavelength_units=self.units_for("wavelengths"),
            spectral_units=self.units_for("spectrum"),
        )

    def down_resolve(self, convolve_factor: int, new_resolution: float | int) -> Self:
        return self.down_bin(new_resolution, spectrum=self.convolve(convolve_factor))
