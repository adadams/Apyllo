from collections.abc import Callable, Sequence
from os import PathLike
from typing import Any, Optional, TypedDict

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pint import Unit, UnitRegistry
from pint_xarray import setup_registry
from xarray import Coordinates, DataArray, Dataset, Variable, apply_ufunc

from user_directories import USER_DIRECTORY

ADDITIONAL_UNITS_FILE = USER_DIRECTORY / "specifications" / "additional_units.txt"


class AttributeBlueprint(TypedDict):
    units: str


class SpecifiedDataBlueprint(TypedDict):
    data: NDArray[np.float_]
    attrs: Optional[AttributeBlueprint]


class VariableBlueprint(TypedDict):
    dims: str | Sequence[str]
    data: NDArray[np.float_]
    attrs: Optional[AttributeBlueprint]
    encoding: Optional[dict[str, Any]]


class DatasetBlueprint(TypedDict):
    data_vars: dict[str, VariableBlueprint]
    coords: Optional[dict[str, VariableBlueprint]]
    attrs: Optional[AttributeBlueprint]


def prep_unit_registry(
    additional_units_file: PathLike = ADDITIONAL_UNITS_FILE,
) -> UnitRegistry:
    unit_registry: UnitRegistry = setup_registry(UnitRegistry())
    unit_registry.load_definitions(additional_units_file)

    return unit_registry


def unit_safe_apply_ufunc(
    function: Callable[[ArrayLike], ArrayLike],
    *args: list[DataArray],
    output_units: dict[str, str | Unit],
    **kwargs: dict[str, Any],
) -> DataArray | tuple[DataArray]:
    unit_safe_args: list[Variable | DataArray] = [
        variable.pint.dequantify() for variable in args
    ]

    applied_function_output: DataArray | tuple[DataArray] = apply_ufunc(
        function, *unit_safe_args, keep_attrs=False, **kwargs
    )

    if isinstance(applied_function_output, tuple):
        for variable, (output_name, output_unit) in zip(
            applied_function_output, output_units.items()
        ):
            variable.rename(output_name)
            variable.attrs["units"] = output_unit

        return tuple(variable.pint.quantify() for variable in applied_function_output)

    elif isinstance(applied_function_output, DataArray):
        output_name, output_unit = next(iter(output_units.items()))

        applied_function_output.rename(output_name)
        applied_function_output.attrs["units"] = output_unit

        return applied_function_output.pint.quantify()

    else:
        raise ValueError(
            "Applied function did not return a DataArray or tuple of DataArrays"
        )


def apply_type_preserving_function(): ...


def format_array_with_specifications(
    data_array: NDArray[np.float_],
    data_format: dict[str, AttributeBlueprint],
) -> dict[str, SpecifiedDataBlueprint]:
    if len(data_format) != len(data_array):
        raise ValueError(
            f"Data format length ({len(data_format)}) does not match "
            f"the number of data entries ({len(data_array)})."
        )

    return {
        variable_name: {"data": variable_data, "attrs": variable_attributes}
        for (variable_name, variable_attributes), variable_data in zip(
            data_format.items(), data_array
        )
    }


def read_specified_data_into_variable(
    specified_data: SpecifiedDataBlueprint,
    dimension_names: str | Sequence[str],
    **encoding_kwargs,
) -> dict[str, Variable]:
    dimension_names: Sequence[str] = (
        [dimension_names] if isinstance(dimension_names, str) else dimension_names
    )

    assert specified_data["data"].ndim == len(
        dimension_names
    ), "Data must have same number of dimensions as dimension names."

    return Variable(**specified_data, dims=dimension_names, **encoding_kwargs)


def add_coordinates_to_individual_variable(
    variable: Variable,
    coordinates: Coordinates,
    variable_name: str = None,
    attach_units: bool = True,
) -> DataArray:
    dimension_names: list[str] = list(coordinates.keys())

    assert all(
        [dimension_name in variable.dims for dimension_name in dimension_names]
    ), "There are dimensions specified in the coordinates that are not in the data."

    variable_with_coordinates: DataArray = DataArray(
        **variable.to_dict(), coords=coordinates, name=variable_name
    )

    return (
        variable_with_coordinates.pint.quantify()
        if attach_units
        else variable_with_coordinates
    )


def make_dataset_from_variables(
    data_variables: dict[str, Variable],
    coordinates: Coordinates,
    **dataset_attributes,
) -> Dataset:
    return Dataset(
        data_vars=data_variables, coords=coordinates, attrs=dataset_attributes
    )
