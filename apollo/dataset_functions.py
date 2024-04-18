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
    """Open, load into memory, and close a Dataset from a file or file-like
    object.

    This is a thin wrapper around :py:meth:`~xarray.open_dataset`. It differs
    from `open_dataset` in that it loads the Dataset into memory, closes the
    file, and returns the Dataset. In contrast, `open_dataset` keeps the file
    handle open and lazy loads its contents. All parameters are passed directly
    to `open_dataset`. See that documentation for further details.

    Returns
    -------
    dataset : Dataset
        The newly created Dataset.

    See Also
    --------
    open_dataset
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
