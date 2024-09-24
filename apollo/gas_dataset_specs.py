from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray
from xarray import Dataset, Variable


def make_pressure_coordinate(
    pressures: NDArray[np.float64],
    pressure_attrs: Optional[dict[str, Any]] = None,
) -> Variable:
    pressure_attrs = pressure_attrs or {"units": "bar"}

    return Variable(
        dims="pressure",
        data=pressures,
        attrs=pressure_attrs,
    )


# Implied function that maps uniform single mixing ratio to gas spec.


def make_gas_spec(log10_mixing_ratio: NDArray[np.float64]) -> Variable:
    return Variable(
        dims=["pressure"],
        data=10**log10_mixing_ratio,
        attrs={"units": "dimensionless"},
    )


def make_gas_spec_dataset(
    gas_specs: dict[str, Variable], pressures: NDArray[np.float64]
) -> Dataset:
    pressure_coordinate: Variable = make_pressure_coordinate(pressures)
    coordinate_dictionary: dict[str, Variable] = {"pressure": pressure_coordinate}

    return Dataset(
        data_vars=gas_specs,
        coords=coordinate_dictionary,
        attrs={"title": "Gas Mixing Ratios"},
    )
