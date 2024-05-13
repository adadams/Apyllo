from os import PathLike
from typing import Any, Callable, Optional, Sequence, TypedDict

from numpy.typing import ArrayLike
from pint import UnitRegistry
from pint_xarray import setup_registry
from xarray import DataArray, Dataset, Variable, open_dataset

from user_directories import USER_DIRECTORY

ADDITIONAL_UNITS_FILE = USER_DIRECTORY / "additional_units.txt"


class AttributeBlueprint(TypedDict):
    units: str


class VariableBlueprint(TypedDict):
    data: ArrayLike
    dims: str | Sequence[str]
    attrs: Optional[AttributeBlueprint]
    encoding: Optional[dict[str, Any]]


class DataArrayBlueprint(TypedDict):
    data: ArrayLike
    dims: str | Sequence[str]
    name: str
    attrs: Optional[AttributeBlueprint]


def prep_unit_registry(additional_units_file: PathLike = ADDITIONAL_UNITS_FILE):
    unit_registry = setup_registry(UnitRegistry())
    unit_registry.load_definitions(additional_units_file)

    return unit_registry


def load_dataset_with_units(
    filename_or_obj: PathLike, units: Sequence[str] = None, **kwargs
) -> Dataset:
    """Effectively the same as load_dataset() in xarray (see the relevant documentation),
    but with automatic loading of units using Pint. One has to store the units as text
    attributes in the netcdf files, and tell Pint to add them to the data arrays when one
    loads them into xarray datasets.
    """
    if units is None:
        units = []

    if "cache" in kwargs:
        raise TypeError("cache has no effect in this context")

    with open_dataset(filename_or_obj, **kwargs) as dataset_IO:
        dataset = dataset_IO.load()

        for variable_name, variable in dataset.data_vars.items():
            # order of precedence:
            # (1) units provided in argument
            # (2) "units" in dataset file attributes
            if variable_name in units:
                variable = variable.assign_attrs(units=units[variable_name])
            elif "units" not in variable.attrs:
                variable = variable.assign_attrs(units="dimensionless")

        dataset_with_units = dataset.pint.quantify(unit_registry=prep_unit_registry())

    return dataset_with_units


def save_dataset_with_units(
    dataset: Dataset, *to_netcdf_args, **to_netcdf_kwargs
) -> None:
    dataset_with_units_as_attrs = dataset.pint.dequantify()

    return dataset_with_units_as_attrs.to_netcdf(*to_netcdf_args, **to_netcdf_kwargs)


def change_units(dataarray: DataArray, new_units: str) -> Dataset:
    return {dataarray.name: dataarray.pint.to(new_units)}


# Older function, assess whether it's needed in the face of other newer functions.
def organize_parameter_data_in_xarray(
    name: str,
    value: float | int,
    unit: str,
    coords: dict[str, Sequence[float]] = None,
    **extra_attributes,
):
    dataarray_construction_kwargs = dict(
        data=value, name=name, attrs=dict(**extra_attributes)
    )

    if coords is not None:
        dataarray_construction_kwargs["coords"] = coords

    unit_registry = prep_unit_registry()
    data_array = DataArray(**dataarray_construction_kwargs).pint.quantify(
        unit, unit_registry
    )

    return data_array


# Just a wrapper for constructing xarray.Variable objects.
def read_data_into_xarray_variable(
    data: ArrayLike,
    dimension_names: str | Sequence[str],
    units: str | Sequence[str],
    name: Optional[str] = None,
    **encoding_kwargs,
):
    attributes = {"units": units, "name": name}

    assert data.ndim == len(
        dimension_names
    ), "Data must have same number of dimensions as dimension names."

    return Variable(
        data=data, dims=dimension_names, attrs=attributes, **encoding_kwargs
    )


def add_dimension_to_xarray_variable(
    variable: Variable, coordinates: Variable | Sequence[Variable]
) -> DataArray:
    dimension_names = (
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
    variable_dictionary: dict[str, ArrayLike],
    dimension_names: Sequence[str],
    attributes: dict[str, AttributeBlueprint] = None,
) -> dict[str, dict[str, Any]]:
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


def get_attribute_from_dataset(
    input_dataset: Dataset, attribute_name: str
) -> list[Any]:
    return [
        dataarray.attrs[attribute_name]
        for variable_name, dataarray in input_dataset.items()
    ]


def extract_dataset_subset_by_attribute(
    dataset: Dataset, attribute_condition: Callable[[Dataset, str], bool]
) -> Dataset:
    return dataset.get(
        [
            data_var
            for data_var in dataset.data_vars
            if attribute_condition(dataset, data_var)
        ]
    )


def extract_dataset_subset_by_parameter_group(
    dataset: Dataset, group_name: str, group_attribute_label: str = "base_group"
) -> Dataset:
    def belongs_to_a_group(dataset, variable):
        return dataset.get(variable).attrs[group_attribute_label] == group_name

    return extract_dataset_subset_by_attribute(dataset, belongs_to_a_group)


def extract_free_parameters_from_dataset(dataset: Dataset) -> Dataset:
    def is_a_free_parameter(dataset, variable):
        return dataset.get(variable).attrs["base_group"] != "Derived"

    return extract_dataset_subset_by_attribute(dataset, is_a_free_parameter)
