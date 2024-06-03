from pathlib import Path
from typing import Any, Sequence

import numpy as np
import tomllib
from numpy.typing import NDArray
from xarray import Dataset

from apollo.convenience_types import Pathlike
from apollo.dataset.dataset_builders import (
    AttributeBlueprint,
    make_dataset_variables_from_dict,
)
from apollo.spectrum.spectral_classes import get_wavelengths_from_wavelength_bins

with open("apollo/formats/APOLLO_data_file_format.toml", "rb") as data_format_file:
    APOLLO_DATA_FORMAT: dict[str, dict[str, Any]] = tomllib.load(data_format_file)


def read_data_array_into_dictionary(
    data_array: NDArray[np.float_], attributes: AttributeBlueprint = None
) -> dict[str, Any]:
    if attributes is None:
        attributes = {}

    variable_dictionary: dict[str, NDArray[np.float_]] = {
        variable_name: data_row
        for variable_name, data_row in zip(attributes.keys(), data_array)
    }
    return make_dataset_variables_from_dict(
        variable_dictionary, "wavelength", attributes
    )


def read_APOLLO_data_into_dictionary(
    filepath: Pathlike, data_format: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    data_as_numpy = np.loadtxt(filepath, dtype=np.float32).T

    return read_data_array_into_dictionary(data_as_numpy, data_format)


def make_wavelength_coordinate_dictionary_from_APOLLO_data_dictionary(
    APOLLO_data_dictionary: dict[str, float],
) -> dict[str, Any]:
    return {
        "dims": "wavelength",
        "data": get_wavelengths_from_wavelength_bins(
            APOLLO_data_dictionary["wavelength_bin_starts"]["data"],
            APOLLO_data_dictionary["wavelength_bin_ends"]["data"],
        ),
        "attrs": APOLLO_data_dictionary["wavelength_bin_starts"]["attrs"],
    }


def read_APOLLO_data_into_dataset(
    filepath: Pathlike,
    data_file_format: dict[str, dict[str, Any]] = APOLLO_DATA_FORMAT,
    data_name: str = None,
    band_names: Sequence[str] = None,
) -> Dataset:
    if data_name is None:
        data_name: str = Path(filepath).stem

    data_dictionary: dict[str, Any] = read_APOLLO_data_into_dictionary(
        filepath, data_file_format
    )

    wavelength_coordinates: dict[str, Any] = (
        make_wavelength_coordinate_dictionary_from_APOLLO_data_dictionary(
            data_dictionary
        )
    )
    coordinates: dict[str, dict[str, Any]] = {"wavelength": wavelength_coordinates}

    if band_names:
        band_slices: list[slice] = find_band_slices_from_wavelength_bins(
            data_dictionary["wavelength_bin_starts"]["data"],
            data_dictionary["wavelength_bin_ends"]["data"],
        )

        band_coordinate_dictionary: dict[str, Any] = (
            make_band_coordinate_dictionary_for_dataset(band_slices, band_names)
        )
        coordinates["band"] = band_coordinate_dictionary

    dataset_dictionary: dict[str, Any] = {
        "attrs": {"title": data_name},
        "coords": coordinates,
        "dims": "wavelength",
        "data_vars": data_dictionary,
    }

    return Dataset.from_dict(dataset_dictionary).pint.quantify()


def find_band_slices_from_wavelength_bins(
    wavelength_bin_starts: Sequence[float], wavelength_bin_ends: Sequence[float]
) -> list[slice]:
    indices_where_bands_end = np.argwhere(
        ~np.isin(wavelength_bin_ends, wavelength_bin_starts)
    ).squeeze()

    indices_bounding_bands = (0, *(indices_where_bands_end + 1))

    band_slices = [
        slice(start, end)
        for start, end in zip(indices_bounding_bands[:-1], indices_bounding_bands[1:])
    ]
    return band_slices


def make_band_coordinate_dictionary_for_dataset(
    band_slices: Sequence[slice], band_names: Sequence[str]
) -> dict[str, Any]:
    assert len(band_slices) == len(
        band_names
    ), "Number of provided data band names does not match the number of slices found."

    band_lengths = [band_slice.stop - band_slice.start for band_slice in band_slices]

    band_array = np.concatenate(
        [
            [band_name] * band_length
            for band_name, band_length in zip(band_names, band_lengths)
        ]
    )

    return {"dims": "wavelength", "data": band_array}
