from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Any, Optional

import numpy as np
import tomllib
from numpy.typing import NDArray
from xarray import Coordinates, Dataset, Variable

from apollo.dataset.builders import (
    AttributeBlueprint,
    DatasetBlueprint,
    SpecifiedDataBlueprint,
    VariableBlueprint,
    format_array_with_specifications,
    read_specified_data_into_variable,
)
from apollo.formats.custom_types import Pathlike
from apollo.spectrum.band_bin_and_convolve import (
    find_band_slices_from_wavelength_bins,
    get_wavelengths_from_wavelength_bins,
)
from apollo.useful_internal_functions import compose

with open("apollo/formats/APOLLO_data_file_format.toml", "rb") as data_format_file:
    APOLLO_DATA_FORMAT: dict[str, AttributeBlueprint] = tomllib.load(data_format_file)


add_specifications_to_APOLLO_data: Callable[
    [NDArray[np.float_]], dict[str, AttributeBlueprint]
] = partial(format_array_with_specifications, data_format=APOLLO_DATA_FORMAT)

load_APOLLO_data_file: Callable[[Pathlike], dict[str, SpecifiedDataBlueprint]] = (
    compose(
        partial(np.loadtxt, dtype=np.float32),
        np.transpose,
        add_specifications_to_APOLLO_data,
    )
)

read_spectral_data_into_variable: Callable[[SpecifiedDataBlueprint], Variable] = (
    partial(read_specified_data_into_variable, dimension_names=["wavelength"])
)


def make_wavelength_coordinate(
    wavelength_bin_starts: Variable,
    wavelength_bin_ends: Variable,
    wavelength_attrs: Optional[dict[str, Any]] = None,
) -> Variable:
    wavelength_attrs = wavelength_attrs or wavelength_bin_starts.attrs

    return Variable(
        dims="wavelength",
        data=get_wavelengths_from_wavelength_bins(
            wavelength_bin_starts=wavelength_bin_starts.values,
            wavelength_bin_ends=wavelength_bin_ends.values,
        ),
        attrs=wavelength_attrs,
    )


def make_band_coordinate(
    band_slices: Sequence[slice], band_names: Sequence[str]
) -> VariableBlueprint:
    assert len(band_slices) == len(band_names), (
        f"Number of provided data band names ({len(band_names)}) "
        f"does not match the number of slices found ({len(band_slices)})."
    )

    band_lengths: tuple[int] = tuple(
        band_slice.stop - band_slice.start for band_slice in band_slices
    )

    band_array: NDArray[np.str_] = np.concatenate(
        [
            [band_name] * band_length
            for band_name, band_length in zip(band_names, band_lengths)
        ]
    )

    return Variable(
        dims="wavelength",
        data=band_array,
        attrs={"bands": tuple(band_names)},
    )


def read_APOLLO_data_into_dataset(
    filepath: Pathlike,
    band_names: Sequence[str] = None,
    data_name: str = None,
    **additional_dataset_attributes: Any,
) -> Dataset:
    data_name = data_name or Path(filepath).stem

    specified_APOLLO_data: dict[str, SpecifiedDataBlueprint] = load_APOLLO_data_file(
        filepath
    )

    data_variables: dict[str, Variable] = {
        variable_name: read_spectral_data_into_variable(variable_with_specs)
        for variable_name, variable_with_specs in specified_APOLLO_data.items()
    }

    wavelength_coordinate: Variable = make_wavelength_coordinate(
        wavelength_bin_starts=data_variables["wavelength_bin_starts"],
        wavelength_bin_ends=data_variables["wavelength_bin_ends"],
    )

    coordinate_dictionary: dict[str, Variable] = {"wavelength": wavelength_coordinate}

    if band_names:
        band_coordinate: Variable = make_band_coordinate(
            band_slices=find_band_slices_from_wavelength_bins(
                wavelength_bin_starts=data_variables["wavelength_bin_starts"],
                wavelength_bin_ends=data_variables["wavelength_bin_ends"],
            ),
            band_names=band_names,
        )

        coordinate_dictionary["band"] = band_coordinate

    coordinates: Coordinates = Coordinates(coordinate_dictionary)

    dataset_inputs: DatasetBlueprint = {
        "data_vars": data_variables,
        "coords": coordinates,
        "attrs": {"title": data_name, **additional_dataset_attributes},
    }

    return Dataset(**dataset_inputs).pint.quantify()
