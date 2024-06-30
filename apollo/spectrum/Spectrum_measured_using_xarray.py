from collections.abc import Callable
from functools import partial
from typing import Self

import numpy as np
from numpy.typing import ArrayLike
from xarray import Coordinates, DataArray, Dataset, Variable

from apollo.dataset.builders import unit_safe_apply_ufunc
from apollo.spectrum.band_bin_and_convolve import (
    BinSpecwithErrors,
    ConvSpec,
    find_band_slices_from_wavelength_bins,
    get_wavelengths_from_wavelength_bins,
)
from apollo.spectrum.read_spectral_data_into_xarray import make_band_coordinate
from apollo.spectrum.Spectrum_using_xarray import Spectrum
from apollo.useful_internal_functions import count_number_of_arguments


class DataSpectrum(Spectrum):
    def __init__(self, data: Dataset) -> Self:
        self._data: Dataset = data

    def __repr__(self) -> str:
        spectrum_print_line: str = (
            f"Spectrum for {self.title}, band(s) {self.bands} {self._data}"
        )
        resolution_print_line: str = (
            f"Pixel resolution: {self.mean_resolution.pint.magnitude:.2f}"
        )
        signal_to_noise_print_line: str = (
            f"S/N per pixel: {self.mean_signal_to_noise.pint.magnitude:.2f}"
        )

        lines: list[str] = [
            spectrum_print_line,
            resolution_print_line,
            signal_to_noise_print_line,
        ]

        return "\n".join(lines)

    @property
    def lower_errors(self):
        return self._data.lower_errors

    @property
    def upper_errors(self):
        return self._data.upper_errors

    @property
    def errors(self):
        return (self.lower_errors + self.upper_errors) / 2

    @property
    def mean_signal_to_noise(self) -> float:
        return np.mean(self._data.flux / self.errors)

    # NOTE: something like T = TypeVar("T", bound=ArrayLike) ?
    def convolve(self, convolve_factor: float | int) -> tuple[DataArray, DataArray]:
        convolving_function: Callable[[ArrayLike], ArrayLike] = partial(
            ConvSpec, bin_width=convolve_factor
        )

        convolved_flux: DataArray = unit_safe_apply_ufunc(
            convolving_function,
            self.flux.pint.dequantify(),
            input_core_dims=[["wavelength"]],
            output_core_dims=[["wavelength"]],
        ).pint.quantify()

        convolved_errors: DataArray = unit_safe_apply_ufunc(
            convolving_function,
            self.errors.pint.dequantify(),
            input_core_dims=[["wavelength"]],
            output_core_dims=[["wavelength"]],
        ).pint.quantify()

        return convolved_flux, convolved_errors

    def down_bin(
        self,
        new_resolution: float | int,
        flux: DataArray = None,
        errors: DataArray = None,
        new_title=None,
    ) -> Self:
        flux = flux if flux is not None else self.flux
        errors = errors if errors is not None else self.errors
        new_title = new_title or f"{self.title}_R{new_resolution}"

        original_resolution: float = self.mean_resolution
        bin_factor: float = original_resolution / new_resolution

        binning_function = partial(BinSpecwithErrors, binw=bin_factor)
        number_of_binning_function_arguments: int = (
            count_number_of_arguments(binning_function) - 1
        )
        number_of_binning_function_outputs: int = number_of_binning_function_arguments

        binned_WBS, binned_WBE, binned_flux, binned_errors = unit_safe_apply_ufunc(
            binning_function,
            self.wavelength_bin_starts,
            self.wavelength_bin_ends,
            flux,
            errors,
            input_core_dims=[["wavelength"]] * number_of_binning_function_arguments,
            output_core_dims=[["wavelength"]] * number_of_binning_function_outputs,
            exclude_dims={"wavelength"},
        )

        number_of_wavelength_arguments: int = count_number_of_arguments(
            get_wavelengths_from_wavelength_bins
        )
        number_of_wavelength_outputs: int = 1  # just hard code for now

        binned_wavelength_coordinate: DataArray = unit_safe_apply_ufunc(
            get_wavelengths_from_wavelength_bins,
            binned_WBS,
            binned_WBE,
            input_core_dims=[["wavelength"]] * number_of_wavelength_arguments,
            output_core_dims=[["wavelength"]] * number_of_wavelength_outputs,
        )

        binned_coordinate_dictionary: dict[str, Variable] = {
            "wavelength": binned_wavelength_coordinate
        }

        if "band" in self._data.coords:
            binned_band_coordinate: Variable = make_band_coordinate(
                band_slices=find_band_slices_from_wavelength_bins(
                    wavelength_bin_starts=binned_WBS, wavelength_bin_ends=binned_WBE
                ),
                band_names=self.bands,
            )

            binned_coordinate_dictionary["band"] = binned_band_coordinate

        binned_coordinates: Coordinates = Coordinates(binned_coordinate_dictionary)

        binned_dataset: Dataset = Dataset(
            data_vars={
                "wavelength_bin_starts": binned_WBS,
                "wavelength_bin_ends": binned_WBE,
                "flux": binned_flux,
                "lower_errors": binned_errors,
                "upper_errors": binned_errors,
            },
            coords=binned_coordinates,
            attrs={"title": new_title},
        )

        return Spectrum(binned_dataset)

    def down_resolve(self, convolve_factor: int, new_resolution: float | int) -> Self:
        return self.down_bin(new_resolution, *self.convolve(convolve_factor))
