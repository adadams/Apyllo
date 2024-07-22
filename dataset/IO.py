from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Final, Sequence

from pint import UnitRegistry
from pint_xarray import setup_registry
from xarray import Dataset, load_dataset

from custom_types import Pathlike
from useful_internal_functions import compose
from user_directories import USER_DIRECTORY

ADDITIONAL_UNITS_FILE: Final[Path] = (
    USER_DIRECTORY / "specifications" / "additional_units.txt"
)

UNSAFE_VARIABLE_NAMES_TO_SAFE: Final[dict[str, str]] = {
    "C/O": "CtoO_ratio",
    "[Fe/H]": "Metallicity",
}

SAFE_VARIABLE_NAMES_TO_UNSAFE: Final[dict[str, str]] = {
    safe_name: unsafe_name
    for unsafe_name, safe_name in UNSAFE_VARIABLE_NAMES_TO_SAFE.items()
}


def check_and_replace_variable_names(
    dataset: Dataset,
    variable_mapping: dict[str, str],
) -> Dataset:
    for initial_name, final_name in variable_mapping.items():
        if initial_name in dataset.data_vars:
            dataset = dataset.rename({initial_name: final_name})

    return dataset


def prep_unit_registry(
    additional_units_file: Pathlike = ADDITIONAL_UNITS_FILE,
) -> UnitRegistry:
    unit_registry: UnitRegistry = setup_registry(UnitRegistry())
    unit_registry.load_definitions(additional_units_file)

    return unit_registry


def put_dataset_units_in_attrs(dataset: Dataset) -> Dataset:
    return dataset.pint.dequantify()


def pull_dataset_units_from_attrs(
    dataset: Dataset, units: Sequence[str] | None = None
) -> Dataset:
    for variable_name, variable in dataset.data_vars.items():
        # order of precedence:
        # (1) units provided in argument
        # (2) "units" in dataset file attributes
        if variable_name in units:
            variable = variable.assign_attrs(units=units[variable_name])
        elif "units" not in variable.attrs:
            variable = variable.assign_attrs(units="dimensionless")

    return dataset.pint.quantify(unit_registry=prep_unit_registry())


load_and_prep_dataset: Callable[[Dataset], Dataset] = compose(
    load_dataset,
    pull_dataset_units_from_attrs,
    partial(
        check_and_replace_variable_names, variable_mapping=SAFE_VARIABLE_NAMES_TO_UNSAFE
    ),
)


prep_dataset_for_saving: Callable[[Dataset], Dataset] = compose(
    put_dataset_units_in_attrs,
    partial(
        check_and_replace_variable_names, variable_mapping=UNSAFE_VARIABLE_NAMES_TO_SAFE
    ),
)


def prep_and_save_dataset(
    dataset: Dataset, *to_netcdf_args, **to_netcdf_kwargs
) -> None:
    prepped_dataset: Dataset = prep_dataset_for_saving(dataset)

    return prepped_dataset.to_netcdf(*to_netcdf_args, **to_netcdf_kwargs)
