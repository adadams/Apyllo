from os import PathLike
from pint import UnitRegistry
from pint_xarray import setup_registry
from typing import Any, Callable, Sequence
from xarray import DataArray, Dataset, open_dataset

from user_directories import USER_DIRECTORY

ADDITIONAL_UNITS_FILE = USER_DIRECTORY / "additional_units.txt"


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
            # order of precedence: (1) units provided in argument, (2) "units" in dataset file attributes
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


def make_dataset_variables_from_dict(
    format_dict, values, dimension_names: Sequence[str]
) -> dict[str, dict[str, Any]]:
    return {
        variable_name: {
            "dims": dimension_names,
            "data": variable_values,
            **extra_attributes,
        }
        for variable_name, variable_values in input_dict.items()
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
    belongs_to_a_group = (
        lambda dataset, variable: dataset.get(variable).attrs[group_attribute_label]
        == group_name
    )
    return extract_dataset_subset_by_attribute(dataset, belongs_to_a_group)


def extract_free_parameters_from_dataset(dataset: Dataset) -> Dataset:
    is_a_free_parameter = (
        lambda dataset, variable: dataset.get(variable).attrs["base_group"] != "Derived"
    )
    return extract_dataset_subset_by_attribute(dataset, is_a_free_parameter)
