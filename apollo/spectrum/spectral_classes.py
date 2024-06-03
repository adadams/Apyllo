import numpy as np
from numpy.typing import NDArray
from xarray import DataArray, Dataset

from apollo.convenience_types import Measured
from apollo.dataset.dataset_builders import make_dataset_variables_from_dict
from apollo.dataset.uses_Xarray import UsesXarray
from apollo.spectrum.bin_and_convolve import BinSpec, ConvSpec
from apollo.spectrum.spectral_protocol import Spectral


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


class Spectrum(Spectral, UsesXarray):
    def __init__(self, data: Dataset):
        self.data: Dataset = data
        self.wavelengths: DataArray = get_wavelengths_from_wavelength_bins(
            self.wavelength_bin_starts, self.wavelength_bin_ends
        )

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
        dataset: Dataset = make_dataset_variables_from_dict(
            {
                "wavelength_bin_starts": wavelength_bin_starts,
                "wavelength_bin_ends": wavelength_bin_ends,
                "wavelength": wavelengths,
                "spectrum": spectrum,
            },
            dimension_names=["wavelength"],
            units={"wavelength": wavelength_units, "spectrum": spectral_units},
        )
        return cls(dataset)

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

    def convolve(self, convolve_factor):
        convolved_spectrum = ConvSpec(self.spectrum, convolve_factor)

        return self.wavelength_bin_starts, self.wavelength_bin_ends, convolved_spectrum

    def down_bin(
        self, wavelength_bin_starts, wavelength_bin_ends, spectrum, new_resolution
    ):
        original_resolution: float = self.mean_resolution
        bin_factor: float | int = original_resolution / new_resolution

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


class DataSpectrum(Spectrum, Measured, UsesXarray):
    def __repr__(self) -> str:
        return (
            f"Spectrum for {self.data.attrs["title"]} {self.data} \n"
            + f"Pixel resolution: {self.mean_resolution.pint.magnitude:.2f} \n"
            + f"S/N per pixel: {self.mean_signal_to_noise.pint.magnitude:.2f}"
        )

    @property
    def errors(self):
        return (self.lower_errors + self.upper_errors) / 2

    @property
    def mean_signal_to_noise(self):
        return np.mean(self.spectrum / self.errors)

    def convolve(self, convolve_factor):
        convolved_spectrum = ConvSpec(self.spectrum, convolve_factor)
        convolved_errors = ConvSpec(self.errors, convolve_factor)

        return (
            self.wavelength_bin_starts,
            self.wavelength_bin_ends,
            convolved_spectrum,
            convolved_errors,
        )

    def down_bin(self, new_resolution):
        original_resolution = self.mean_resolution
        bin_factor = original_resolution / new_resolution

        # Bin the observations to fit a lower sampling resolution
        binned_spectrum, binned_errors, binned_WBS, binned_WBE = BinSpec(
            self.spectrum,
            self.errors,
            self.wavelength_bin_starts,
            self.wavelength_bin_ends,
            bin_factor,
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
