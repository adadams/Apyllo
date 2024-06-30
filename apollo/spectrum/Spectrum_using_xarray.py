from collections.abc import Callable
from functools import partial
from typing import Self

import numpy as np
from numpy.typing import ArrayLike
from xarray import Coordinates, DataArray, Dataset, Variable, register_dataset_accessor

from apollo.dataset.builders import unit_safe_apply_ufunc
from apollo.spectrum.band_bin_and_convolve import (
    BinSpec,
    ConvSpec,
    find_band_slices_from_wavelength_bins,
    get_wavelengths_from_wavelength_bins,
)
from apollo.spectrum.read_spectral_data_into_xarray import make_band_coordinate
from apollo.useful_internal_functions import count_number_of_arguments


@register_dataset_accessor("spectrum")
class Spectrum:
    def __init__(self, data: Dataset) -> Self:
        self._data: Dataset = data

    def __repr__(self) -> str:
        spectrum_print_line: str = (
            f"Spectrum for {self.title}, band(s) {self.bands} {self._data}"
        )
        resolution_print_line: str = f"Pixel resolution: {self.mean_resolution:.2f}"

        lines: list[str] = [spectrum_print_line, resolution_print_line]

        return "\n".join(lines)

    @property
    def title(self) -> str:
        return self._data.attrs["title"]

    @property
    def bands(self) -> set[str]:
        return set(self._data.band.values)

    @property
    def wavelength_bin_starts(self) -> DataArray:
        return self._data.get("wavelength_bin_starts")

    @property
    def wavelength_bin_ends(self) -> DataArray:
        return self._data.get("wavelength_bin_ends")

    @property
    def wavelengths(self) -> DataArray:
        return self._data.get("wavelength")

    @property
    def flux(self) -> DataArray:
        return self._data.get("flux")

    @property
    def pixel_resolution(self) -> DataArray:
        return self.wavelengths / (
            self.wavelength_bin_ends - self.wavelength_bin_starts
        )

    @property
    def mean_resolution(self, pixels_per_resolution_element=1) -> float:
        return np.mean(self.pixel_resolution.values / pixels_per_resolution_element)

    def units_for(self, variable_name: str) -> str:
        variable: DataArray = self._data.get(variable_name)
        return (
            variable.attrs["units"]
            if "units" in self._data.get(variable_name)
            else variable.pint.units
        )

    def convolve(self, convolve_factor: float | int) -> DataArray:
        convolving_function: Callable[[ArrayLike], ArrayLike] = partial(
            ConvSpec, bin_width=convolve_factor
        )

        return unit_safe_apply_ufunc(
            convolving_function,
            self.flux,
            input_core_dims=[["wavelength"]],
            output_core_dims=[["wavelength"]],
        )

    def down_bin(self, new_resolution, flux=None, new_title=None) -> Self:
        flux = flux if flux is not None else self.flux
        new_title = new_title or f"{self.title}_R{new_resolution}"

        original_resolution: float = self.mean_resolution
        bin_factor: float | int = original_resolution / new_resolution

        binning_function = partial(BinSpec, binw=bin_factor)
        number_of_binning_function_arguments: int = count_number_of_arguments(
            binning_function
        )
        number_of_binning_function_outputs: int = number_of_binning_function_arguments

        binned_WBS, binned_WBE, binned_flux = unit_safe_apply_ufunc(
            binning_function,
            self.wavelength_bin_starts,
            self.wavelength_bin_ends,
            flux,
            input_core_dims=[["wavelength"] * number_of_binning_function_arguments],
            output_core_dims=[["wavelength"] * number_of_binning_function_outputs],
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
            input_core_dims=[["wavelength"] * number_of_wavelength_arguments],
            output_core_dims=[["wavelength"] * number_of_wavelength_outputs],
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
            },
            coords=binned_coordinates,
            attrs={"title": new_title},
        )

        return Spectrum(binned_dataset)

    def down_resolve(self, convolve_factor: int, new_resolution: float | int) -> Self:
        return self.down_bin(new_resolution, self.convolve(convolve_factor))
