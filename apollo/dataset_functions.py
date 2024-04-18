from pathlib import Path
from pint_xarray import unit_registry as ureg
from typing import BinaryIO, Sequence
from xarray import DataArray, Dataset, open_dataset

from apollo.general_protocols import Pathlike
from user_directories import USER_DIRECTORY

ADDITIONAL_UNITS_FILE = USER_DIRECTORY / "additional_units.txt"
if not ureg:
    ureg.load_definitions(ADDITIONAL_UNITS_FILE)


def load_dataset_with_units(
    filename_or_obj: Pathlike | BinaryIO, units: Sequence[str] = None, **kwargs
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

        dataset_with_units = dataset.pint.quantify()

    return dataset_with_units


def save_dataset_with_units(
    dataset: Dataset, *to_netcdf_args, **to_netcdf_kwargs
) -> None:
    dataset_with_units_as_attrs = dataset.pint.dequantify()

    return dataset_with_units_as_attrs.to_netcdf(*to_netcdf_args, **to_netcdf_kwargs)


def change_units(dataarray: DataArray, new_units: str) -> Dataset:
    return {dataarray.name: dataarray.pint.to(new_units)}
