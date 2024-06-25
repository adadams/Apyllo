from os import PathLike
from typing import Any, Optional, Sequence, TypedDict

import numpy as np
from numpy.typing import NDArray
from pint import UnitRegistry
from pint_xarray import setup_registry
from xarray import DataArray, Variable

from user_directories import USER_DIRECTORY

ADDITIONAL_UNITS_FILE = USER_DIRECTORY / "specifications" / "additional_units.txt"


class AttributeBlueprint(TypedDict):
    units: str


class VariableBlueprint(TypedDict):
    dims: str | Sequence[str]
    data: NDArray[np.float_]
    attrs: Optional[AttributeBlueprint]
    encoding: Optional[dict[str, Any]]


class DataArrayBlueprint(TypedDict):
    data: NDArray[np.float_]
    coords: Optional[dict[str, Sequence[float]]]
    dims: str | Sequence[str]
    name: str
    attrs: Optional[AttributeBlueprint]


class DatasetBlueprint(TypedDict):
    data_vars: dict[str, VariableBlueprint]
    coords: Optional[dict[str, Sequence[float]]]
    attrs: Optional[AttributeBlueprint]


def prep_unit_registry(
    additional_units_file: PathLike = ADDITIONAL_UNITS_FILE,
) -> UnitRegistry:
    unit_registry: UnitRegistry = setup_registry(UnitRegistry())
    unit_registry.load_definitions(additional_units_file)

    return unit_registry


# Older function, assess whether it's needed in the face of other newer functions.
def organize_parameter_data_in_xarray(
    name: str,
    value: float | int,
    unit: str,
    coords: dict[str, Sequence[float]] = None,
    **extra_attributes,
) -> DataArray:
    dataarray_construction_kwargs: dict[str, Any] = {
        "data": value,
        "name": name,
        "attrs": dict(**extra_attributes),
    }

    if coords is not None:
        dataarray_construction_kwargs["coords"] = coords

    unit_registry: UnitRegistry = prep_unit_registry()
    data_array: DataArray = DataArray(**dataarray_construction_kwargs).pint.quantify(
        unit, unit_registry
    )

    return data_array


# Just a wrapper for constructing xarray.Variable objects.
def read_data_into_xarray_variable(
    data: NDArray[np.float_],
    dimension_names: str | Sequence[str],
    units: str | Sequence[str],
    name: Optional[str] = None,
    **encoding_kwargs,
) -> Variable:
    attributes: AttributeBlueprint = {"units": units, "name": name}

    assert data.ndim == len(
        dimension_names
    ), "Data must have same number of dimensions as dimension names."

    return Variable(
        data=data, dims=dimension_names, attrs=attributes, **encoding_kwargs
    )


def add_dimension_to_xarray_variable(
    variable: Variable, coordinates: Variable | Sequence[Variable]
) -> DataArray:
    dimension_names: tuple[str] = (
        coordinates.attrs["name"]
        if isinstance(coordinates, Variable)
        else [coordinate.attrs["name"] for coordinate in coordinates]
    )
    assert all(
        [dimension_name in variable.dims for dimension_name in dimension_names]
    ), "There are dimensions specified in the coordinates that are not in the data."

    coordinates_as_dictionary: dict[str, Variable] = {
        dimension_name: coordinate
        for dimension_name, coordinate in zip(dimension_names, coordinates)
    }

    return DataArray(**variable.to_dict(), coords=coordinates_as_dictionary)


def make_dataset_variables_from_dict(
    variable_dictionary: dict[str, NDArray[np.float_]],
    dimension_names: Sequence[str],
    attributes: dict[str, AttributeBlueprint] = None,
) -> dict[str, DataArrayBlueprint]:
    if attributes is None:
        attributes = {variable_name: {} for variable_name in variable_dictionary}

    return {
        variable_name: {
            "dims": dimension_names,
            "data": variable_values,
            "attrs": attribute_values,
        }
        for (variable_name, variable_values), (
            variable_name,
            attribute_values,
        ) in zip(variable_dictionary.items(), attributes.items())
    }
