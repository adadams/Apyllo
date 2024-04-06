from dataclasses import dataclass
from functools import partial
from numpy.typing import NDArray
import numpy as np
from pathlib import Path
from pint import Quantity, UnitRegistry
from typing import Any, Callable, IO, NamedTuple, Optional, Protocol, Sequence
from xarray import Dataset, DataArray
from pint_xarray import unit_registry as ureg

Pathlike = Path | str
Filelike = Pathlike | IO

LIST_OF_MEASURED_VARIABLES = [
    "lower_errors",
    "upper_errors",
]


class Measured(Protocol):
    """A thing that has errors.

    Args:
        Protocol (_type_): _description_

    Returns:
        _type_: _description_
    """

    lower_errors: NDArray
    upper_errors: NDArray

    @property
    def errors(self):
        return (self.lower_errors + self.upper_errors) / 2


class Parametrized(Protocol):
    """
    A sub-model for some component (for example, T-P profile, clouds) that
    has a function that takes parameters and returns some physical ``profile''
    function, i.e. a temperature function that takes pressures directly.
    """

    model_function: Callable

    def generate_profile_function(self, *args, **kwargs) -> None: ...


ADDITIONAL_UNITS_FILE = Path.cwd() / "additional_units.txt"
ureg.load_definitions(ADDITIONAL_UNITS_FILE)


@dataclass
class ModelBuilder:
    """Docstring goes here."""

    """
    You could have
    - a core function, that translates a set of values and some core component of the
        physical structure (pressure, wavelength) that are arguments.
        You return some physical representation of a part of the model.

    - a method that takes the parameters and partially applies them to
        the function, leaving a function of just pressure, for example.

    - a container for plotting information, like the long name, units, etc.

    - flexibility to calculate 
    """
    model_function: Callable
    model_attributes: NamedTuple

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.model_function(*args, **kwargs)

    def load_function(self, *parameter_args, **parameter_kwargs):
        return partial(self.model_function, *parameter_args, **parameter_kwargs)


def organize_parameter_data_in_xarray(
    name: str,
    print_name: str,
    value: float | int,
    unit: str,
    coords: dict[str, Sequence[float]] = None,
    **other_info,
):
    dataarray_construction_kwargs = dict(
        data=value, name=name, attrs=dict(print_name=print_name, **other_info)
    )

    if coords is not None:
        dataarray_construction_kwargs["coords"] = coords

    data_array = DataArray(**dataarray_construction_kwargs).pint.quantify(unit)

    return data_array


@dataclass
class XarrayParameter:
    data: Dataset | DataArray

    @classmethod
    def from_measurement_and_unit(
        cls, *, name: str, print_name: str, value: float | int, unit: str, **other_info
    ):
        data_array = DataArray(data=value).pint.quantify(unit)
        data_array = data_array.assign_attrs(
            name=name, print_name=print_name, **other_info
        )

        return cls(name=name, print_name=print_name, data=data_array)

    def __getattr__(self, __name: str) -> Any:
        return self.data.__getattribute__(__name)

    # def __setattr__(self, __name: str, __value: Any) -> None:
    #    return self.data.__setattr__(__name, __value)

    def __repr__(self):
        return f"{self.name}: {self.data}"


@dataclass
class Parameter:
    name: str
    print_name: str
    quantity: Quantity

    @classmethod
    def from_measurement_and_unit(
        cls, *, name: str, print_name: str, value: float | int, unit: str
    ):
        unit_registry = UnitRegistry()
        unit_registry.load_definitions(ADDITIONAL_UNITS_FILE)
        quantity = value * unit_registry(unit)

        return cls(name, print_name, quantity)

    @classmethod
    def from_parameter(
        cls, *, parameter: Parametrized, new_value, new_unit: Optional[str] = None
    ):
        if new_unit is None:
            new_unit = parameter.units
        return cls.from_measurement_and_unit

    def __post_init__(self, value, unit, print_formatter):
        unit_registry = UnitRegistry()
        unit_registry.load_definitions(ADDITIONAL_UNITS_FILE)
        self._quantity = value * unit_registry(unit)
        self._quantity.default_format = f"{print_formatter}P"

    def __getattr__(self, __name: str) -> Any:
        return self._quantity.__getattribute__(__name)

    def __setattr__(self, __name: str, __value: Any) -> None:
        return self._quantity.__setattr__(__name, __value)

    def __repr__(self):
        return f"{self.name}: {self._quantity}"


@dataclass
class ThermalModel(Parametrized):
    """Returns a profile of T(P) when called."""

    pass


@dataclass
class VerticalModel(Parametrized):
    """Returns a profile of z(P) when called."""

    pass


@dataclass
class CompositionModel(Parametrized):
    """
    Holds models that can return absorption and scattering coefficients
    as functions of T, P or z. And quantities of specific elements???
    I think you want a general object that can give you dTau/dP, and
    then maybe an extension for objects with chemical compositions.
    """

    pass


@dataclass
class AtmosphericModel(Parametrized):
    """
    Returns a profile of optical parameters
    (optical depth: tau, scattering asymmetry: g, single-scattering albedo: omega_0)
    as functions of P/z.
    """

    thermal_model: ThermalModel
    vertical_model: VerticalModel
    composition_models: dict[str, CompositionModel]
