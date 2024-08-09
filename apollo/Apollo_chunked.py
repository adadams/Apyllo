import sys
from collections.abc import Callable
from os.path import abspath
from typing import NamedTuple

from apollo.Apollo_Planet_GetModel import get_fractional_cloud_spectrum, get_spectrum
from apollo.Apollo_Planet_SetParameters import (
    Params1Blueprint,
    make_params1,
    set_parameters,
)
from custom_types import Pathlike

APOLLO_DIRECTORY = abspath(
    "/Users/arthur/Documents/Astronomy/2019/Retrieval/Code/APOLLO"
)
if APOLLO_DIRECTORY not in sys.path:
    sys.path.append(APOLLO_DIRECTORY)

from apollo.Apollo_ProcessInputs import (  # noqa: E402
    ParameterValue,
    set_model_wavelengths_from_opacity_tables_and_data,
    set_Teff_calculation_wavelengths_from_opacity_tables,
    set_up_PyPlanet,
)
from apollo.Apollo_ReadInputsfromFile import (  # noqa: E402
    APOLLOFileReadinBlueprint,
    CloudReadinParameters,
    DataParameters,
    FundamentalReadinParameters,
    GasReadinParameters,
    ModelParameters,
    MolecularParameters,
    OpacityParameters,
    PressureParameters,
    TPReadinParameters,
    TransitParameters,
    get_cloud_deck_log_pressure,
    read_inputs_from_file,
)
from apollo.src.wrapPlanet import PyPlanet  # noqa: E402


def generate_emission_spectrum_from_APOLLO_file(
    input_file: Pathlike,
) -> dict:
    readin_specs: APOLLOFileReadinBlueprint = read_inputs_from_file(input_file)

    parameter_values: list[float] = readin_specs["parameters"][
        "readin_parameters"
    ].plparams

    model_parameters: ModelParameters = readin_specs["settings"]["model_parameters"]

    data_parameters: DataParameters = readin_specs["settings"]["data_parameters"]
    opacity_parameters: OpacityParameters = readin_specs["settings"][
        "opacity_parameters"
    ]
    transit_parameters: TransitParameters = readin_specs["settings"][
        "transit_parameters"
    ]
    pressure_parameters: PressureParameters = readin_specs["settings"][
        "pressure_parameters"
    ]

    model_wavelengths = set_model_wavelengths_from_opacity_tables_and_data(
        data_parameters=data_parameters,
        opacity_parameters=opacity_parameters,
        minDL=0.0,
        maxDL=0.0,  # TODO: add deltaL parameter values here
    )

    TP_readin_parameters: TPReadinParameters = readin_specs["parameters"][
        "TP_readin_parameters"
    ]
    TP_functional_type: str = TP_readin_parameters.atmtype
    TP_function_parameters: list[ParameterValue] = (
        TP_readin_parameters.bodge_TP_parameters(parameter_values)
    )
    Teff_calculation_model_wavelengths = (
        set_Teff_calculation_wavelengths_from_opacity_tables(
            opacdir=opacity_parameters.opacdir
        )
    )

    cloud_readin_parameters: CloudReadinParameters = readin_specs["parameters"][
        "cloud_readin_parameters"
    ]
    cloud_model_mode: int = cloud_readin_parameters.cloudmod
    cloud_haze_type: int = cloud_readin_parameters.hazetype
    cloud_filling_fraction: float = cloud_readin_parameters.get_cloud_filling_fraction(
        parameter_values
    )

    gas_readin_parameters: GasReadinParameters = readin_specs["parameters"][
        "gas_readin_parameters"
    ]
    list_of_gas_species: list[str] = gas_readin_parameters.gases
    gas_parameters: list[ParameterValue] = gas_readin_parameters.bodge_gas_parameters(
        parameter_values
    )
    molecular_parameters: MolecularParameters = (
        gas_readin_parameters.get_molecular_weights_and_scattering_opacities(
            parameter_values
        )
    )

    fundamental_readin_parameters: FundamentalReadinParameters = readin_specs[
        "parameters"
    ]["fundamental_readin_parameters"]
    distance_to_system_in_parsecs: float = readin_specs["settings"][
        "location_parameters"
    ].dist
    radius_parameter: ParameterValue = (
        fundamental_readin_parameters.bodge_radius_parameter(
            parameter_values, distance_to_system_in_parsecs
        )
    )
    gravity_parameter: ParameterValue = (
        fundamental_readin_parameters.bodge_gravity_parameter(parameter_values)
    )

    cloud_readin_parameters: CloudReadinParameters = readin_specs["parameters"][
        "cloud_readin_parameters"
    ]
    cloud_parameter_tuple: NamedTuple = (
        cloud_readin_parameters.make_cloud_parameter_tuple(parameter_values)
    )
    cloud_deck_log_pressure: float = get_cloud_deck_log_pressure(
        cloud_readin_parameters.bodge_cloud_parameters(parameter_values)
    )

    planet: PyPlanet = set_up_PyPlanet(
        model_parameters=model_parameters,
        model_wavelengths=model_wavelengths,
        Teff_calculation_wavelengths=Teff_calculation_model_wavelengths,
        atmtype=TP_functional_type,
        cloudmod=cloud_model_mode,
        hazetype=cloud_haze_type,
        list_of_gas_species=list_of_gas_species,
        opacity_parameters=opacity_parameters,
    )

    params1: Params1Blueprint = make_params1(
        radius_parameter=radius_parameter,
        gravity_parameter=gravity_parameter,
        cloud_deck_log_pressure=cloud_deck_log_pressure,
        cloud_parameters=cloud_parameter_tuple,
        molecular_parameters=molecular_parameters,
        pressure_parameters=pressure_parameters,
        transit_parameters=transit_parameters,
    )

    parameters_for_planet: list = set_parameters(
        params1=params1,
        molecular_parameters=molecular_parameters,
        gas_parameters=gas_parameters,
        TP_model_name=TP_functional_type,
        TP_model_parameters=TP_function_parameters,
        pressure_parameters=pressure_parameters,
    )

    planet.set_Params(*parameters_for_planet)

    model_spectrum_fraction: Callable = (
        get_spectrum if cloud_filling_fraction == 1.0 else get_fractional_cloud_spectrum
    )

    return model_spectrum_fraction(planet)
