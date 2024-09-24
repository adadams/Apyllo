import sys
from collections.abc import Callable
from functools import partial
from os.path import abspath
from typing import Final, NamedTuple, Optional, Union

import numpy as np
from numpy.typing import NDArray

APOLLO_DIRECTORY = abspath(
    "/Users/arthur/Documents/Astronomy/2019/Retrieval/Code/Apyllo"
)
if APOLLO_DIRECTORY not in sys.path:
    sys.path.append(APOLLO_DIRECTORY)

from apollo.Apollo_ReadInputsfromFile import (  # noqa: E402
    MolecularParameters,
    ParameterValue,
    PressureParameters,
    TransitParameters,
)
from apollo.submodels import TP  # noqa: E402

# REARTH_IN_CM = R_earth.to(cm).value
REARTH_IN_CM: Final[float] = 6.371e8

# RJUPITER_IN_REARTH = (R_jup/R_earth).decompose().value
RJUPITER_IN_REARTH: Final[float] = 11.2

PARSEC_IN_REARTH: Final[float] = 4.838e9


class MakeParams1Blueprint(NamedTuple):
    radius_parameter: ParameterValue
    gravity_parameter: ParameterValue
    cloud_deck_parameter: ParameterValue
    cloud_parameters: Optional[NamedTuple]
    transit_parameters: TransitParameters
    molecular_parameters: MolecularParameters
    pressure_parameters: PressureParameters


class Params1Blueprint(NamedTuple):
    radius: float
    log_gravity: float
    cloud_deck_log_pressure: float
    stellar_temperature: float
    stellar_radius: float
    mean_mmw: float
    mean_rxsec: float
    minimum_log_pressure: float
    maximum_log_pressure: float
    semimajor_axis: float


def make_params1(
    radius_parameter: ParameterValue,
    gravity_parameter: ParameterValue,
    cloud_deck_log_pressure: float,
    cloud_parameters: Union[NamedTuple, None],
    molecular_parameters: MolecularParameters,
    pressure_parameters: PressureParameters,
    transit_parameters: TransitParameters,
) -> list[float]:
    mean_molecular_weight: float = np.mean(
        molecular_parameters.weighted_molecular_weights
    )

    mean_scattering_cross_section: float = np.mean(
        molecular_parameters.weighted_scattering_cross_sections
    )

    params1_minus_clouds: Params1Blueprint = Params1Blueprint(
        radius=radius_parameter.value / REARTH_IN_CM,
        log_gravity=gravity_parameter.value,
        cloud_deck_log_pressure=cloud_deck_log_pressure,
        mean_mmw=mean_molecular_weight,
        mean_rxsec=mean_scattering_cross_section,
        minimum_log_pressure=pressure_parameters.minP,
        maximum_log_pressure=pressure_parameters.maxP,
        stellar_temperature=transit_parameters.tstar,
        stellar_radius=transit_parameters.rstar,
        semimajor_axis=transit_parameters.sma,
    )

    result: list[float] = (
        list(params1_minus_clouds)
        if cloud_parameters is None
        else list(params1_minus_clouds) + list(cloud_parameters)
    )
    print(f"make_params1: {result=}")

    return result


def make_abund(free_gas_log_abundances: NDArray[np.float64]) -> NDArray[np.float64]:
    if len(free_gas_log_abundances) == 0:
        return [1.0]

    else:
        free_gas_abundances: NDArray[np.float64] = 10**free_gas_log_abundances
        free_abundance_sum: float = np.sum(free_gas_abundances)

        filler_abundance: float = 1.0 - free_abundance_sum

        # mmw, rxsec = GetScaOpac_linear(free_gas_names, free_gas_abundances)

    return np.r_[filler_abundance, free_gas_log_abundances]


def make_pressures_evenly_log_spaced(
    num_layers_final: int, minimum_log_pressure: float, maximum_log_pressure: float
) -> NDArray[np.float64]:
    return np.linspace(minimum_log_pressure, maximum_log_pressure, num=num_layers_final)


def get_TP_profile_function(
    TP_model_name: str, pressure_parameters: PressureParameters
) -> TP.isTPFunction:
    # TP_model_name: str = (
    #    TP_model_name if TP_model_name != "piette" else "modified_piette"
    # )  # NOTE: bodge for now
    TP_profile_function: TP.isTPFunction = getattr(TP, TP_model_name)
    # assert issubclass(
    #    TP_profile_function, TP.isTPFunction
    # ), f"{TP_profile_function} does not follow the TP.isTPFunction protocol."

    pressures: NDArray[np.float64] = make_pressures_evenly_log_spaced(
        num_layers_final=pressure_parameters.vres,
        minimum_log_pressure=pressure_parameters.minP - 6,
        maximum_log_pressure=pressure_parameters.maxP - 6,
    )

    TP_profile_function: Callable[[NDArray[np.float64]], NDArray[np.float64]] = partial(
        TP_profile_function, pressures=pressures
    )

    return TP_profile_function


# NOTE: This function is currently not used. We can try to incorporate it
# once we fold in the opacity data structures that tell us the temperature bounds.
def clip_TP_profile_to_opacity_limits(
    tplong: NDArray[np.float64], opacity_T_min: float, opacity_T_max: float
) -> NDArray[np.float64]:
    return np.clip(tplong, opacity_T_min, opacity_T_max)


class SetParamsBlueprint(NamedTuple):
    params1: NDArray[np.float64]
    abund: NDArray[np.float64]
    rxsec: NDArray[np.float64]
    tplong: NDArray[np.float64]


def compile_Cclass_parameters(
    params1: Params1Blueprint,
    molecular_parameters: MolecularParameters,
    gas_parameters: list[ParameterValue],
    TP_model_name: str,
    TP_model_parameters: list[ParameterValue],
    pressure_parameters: PressureParameters,
) -> SetParamsBlueprint:
    log_nonfiller_abundances: NDArray[np.float64] = np.array(
        [gas_parameter.value for gas_parameter in gas_parameters]
    )

    weighted_scattering_cross_sections: NDArray[np.float64] = (
        molecular_parameters.weighted_scattering_cross_sections
    )

    TP_function: TP.isTPFunction = get_TP_profile_function(
        TP_model_name=TP_model_name, pressure_parameters=pressure_parameters
    )

    TP_args: list[float] = [
        TP_model_parameter.value for TP_model_parameter in TP_model_parameters
    ]

    TP_profile: NDArray[np.float64] = TP_function(*TP_args)

    result = {
        "params1": list(params1),
        "abund": make_abund(log_nonfiller_abundances),
        "rxsec": weighted_scattering_cross_sections,
        "tplong": TP_profile,
    }

    return result


def set_parameters(
    params1: Params1Blueprint,
    molecular_parameters: MolecularParameters,
    gas_parameters: list[ParameterValue],
    TP_model_name: str,
    TP_model_parameters: list[ParameterValue],
    pressure_parameters: PressureParameters,
) -> list[NDArray[np.float64]]:
    set_params_kwargs: SetParamsBlueprint = compile_Cclass_parameters(
        params1=params1,
        molecular_parameters=molecular_parameters,
        gas_parameters=gas_parameters,
        TP_model_name=TP_model_name,
        TP_model_parameters=TP_model_parameters,
        pressure_parameters=pressure_parameters,
    )

    return [
        set_params_kwargs["params1"],
        set_params_kwargs["abund"],
        set_params_kwargs["rxsec"],
        set_params_kwargs["tplong"],
    ]
