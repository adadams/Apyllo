import sys
from enum import StrEnum, auto
from logging import warn
from os.path import abspath
from typing import Final

import numpy as np
from numpy.typing import NDArray

from apollo.Apollo_ProcessInputs import RetrievalParameter
from apollo.src import ApolloFunctions as af

APOLLO_DIRECTORY = abspath(
    "/Users/arthur/Documents/Astronomy/2019/Retrieval/Code/APOLLO"
)
if APOLLO_DIRECTORY not in sys.path:
    sys.path.append(APOLLO_DIRECTORY)

from apollo.calculate_ctoo_and_metallicity import (  # noqa: E402
    calculate_CtoO_and_metallicity,
)

# REARTH_IN_CM = R_earth.to(cm).value
REARTH_IN_CM: Final[float] = 6.371e8

# RJUPITER_IN_REARTH = (R_jup/R_earth).decompose().value
RJUPITER_IN_REARTH: Final[float] = 11.2

PARSEC_IN_REARTH: Final[float] = 4.838e9


class RadiusInputs(StrEnum):
    Rad = auto()
    RtoD = auto()
    RtoD2U = auto()
    norad = auto()


def set_params():
    pass


def make_params1(number_of_params1_arguments: int):
    params1 = np.zeros(number_of_params1_arguments)

    # params1[0] -> radius in Earth radii

    # Radius handling
    if "Rad" in basic:
        pos = basic.index("Rad")
        params1[0] = x[b1 + pos]
        radius = REARTH_IN_CM * x[b1 + pos]
    elif "RtoD" in basic:
        pos = basic.index("RtoD")
        params1[0] = (
            10 ** x[b1 + pos] * dist * PARSEC_IN_REARTH
        )  # convert R/D to Earth radii
        radius = 10 ** x[b1 + pos] * dist * PARSEC_IN_REARTH * REARTH_IN_CM
    elif "RtoD2U" in basic:
        pos = basic.index("RtoD2U")
        params1[0] = np.sqrt(x[b1 + pos])
        radius = np.sqrt(x[b1 + pos]) * REARTH_IN_CM
    else:
        global norad
        norad = True
        params1[0] = RJUPITER_IN_REARTH
        radius = RJUPITER_IN_REARTH * REARTH_IN_CM
        # Default radius = Jupiter

    # Gravity handling
    if "Log(g)" in basic:
        pos = basic.index("Log(g)")
        params1[1] = x[b1 + pos]
        grav = 10 ** x[b1 + pos]
    else:
        params1[1] = 4.1
        grav = 4.1  # ???

    # Cloud deck handling
    if "Cloud_Base" in clouds:
        pos = clouds.index("Cloud_Base")
        params1[2] = x[c1 + pos]
    elif "P_cl" in clouds:
        pos = clouds.index("P_cl")
        params1[2] = x[c1 + pos]
    elif "Cloud_Base" in basic:
        pos = basic.index("Cloud_Base")
        params1[2] = x[b1 + pos]
    elif "P_cl" in basic:
        pos = basic.index("P_cl")
        params1[2] = x[b1 + pos]
    else:
        params1[2] = 8.5
        # Default cloudless
    if params1[2] < minP:
        params1[2] = minP + 0.01
    # Ensures the cloud deck is inside the model bounds.

    mass = grav * radius * radius / 6.67e-8 / 1.898e30

    params1[3] = tstar
    params1[4] = rstar
    params1[5] = np.sum(mmw)
    params1[6] = np.sum(rxsec)
    params1[7] = minP
    params1[8] = maxP
    params1[9] = sma

    if hazetype != 0:
        if cloudmod == 2 or cloudmod == 3:
            for i in range(0, 4):
                params1[i + 10] = x[c1 + i]
            params1[11] = params1[11]
            params1[12] = params1[12] + 6.0
        if cloudmod == 2:
            params1[13] = params1[12] + params1[13]

    if cloudmod == 4:
        for i in range(0, 5):
            params1[i + 10] = x[c1 + i]
        params1[11] = params1[11] + 6.0


def set_radius(
    radius_case: RadiusInputs, size_parameter: RetrievalParameter, dist: float
) -> float:
    # Radius handling
    if radius_case == RadiusInputs.Rad:
        return REARTH_IN_CM * size_parameter.value
    elif radius_case == RadiusInputs.RtoD:
        return 10**size_parameter.value * dist * PARSEC_IN_REARTH * REARTH_IN_CM
    elif radius_case == RadiusInputs.RtoD2U:
        return np.sqrt(size_parameter.value) * REARTH_IN_CM
    else:
        global norad
        norad = True
        return RJUPITER_IN_REARTH * REARTH_IN_CM


def set_log_gravity(log_gravity: RetrievalParameter = None) -> float:
    DEFAULT_LOGG: Final[float] = 4.1

    return log_gravity if log_gravity is not None else DEFAULT_LOGG


def get_cloud_deck_log_pressure(
    cloud_parameters: dict[str, RetrievalParameter],
    maximum_log_pressure_in_CGS: float = 8.5,
) -> float:
    if "Cloud_Base" in cloud_parameters:
        return cloud_parameters["Cloud_Base"].value

    elif "P_cl" in cloud_parameters:
        return cloud_parameters["P_cl"].value

    else:
        warn(
            "Cloud deck pressure not found under either of expected names: 'Cloud_Base' or 'P_cl'."
            + f"Setting to {10**(maximum_log_pressure_in_CGS-6)} bar."
        )

        return maximum_log_pressure_in_CGS


def check_cloud_deck_log_pressure(
    cloud_deck_log_pressure: float,
    minimum_log_pressure_in_CGS: float = 0.0,
) -> float:
    # Ensures the cloud deck is inside the model bounds.
    return (
        cloud_deck_log_pressure
        if cloud_deck_log_pressure >= minimum_log_pressure_in_CGS
        else minimum_log_pressure_in_CGS + 0.01
    )


class CloudModel(StrEnum):
    no_clouds = auto()
    opaque_deck = auto()
    single_particle_size_uniform_number_density = auto()
    single_particle_size_gaussian_number_density = auto()
    power_law_opacity = auto()


def set_cloud_parameters(
    cloud_model_case: CloudModel, cloud_parameters: dict[str, RetrievalParameter]
) -> list[float]:
    cloud_pressure_parameter_names: tuple[str] = (
        "Haze_minP",
        "Haze_meanP",
        "Haze_maxP",
    )

    cloud_parameter_dict: dict[str, float] = {
        parameter_name: parameter.value + 6
        if parameter_name in cloud_pressure_parameter_names
        else parameter.value
        for parameter_name, parameter in cloud_parameters.items()
    }

    if cloud_model_case == CloudModel.single_particle_size_uniform_number_density:
        cloud_parameter_dict["Haze_maxP"] = cloud_parameter_dict[
            "Haze_minP"
        ] + cloud_parameter_dict.pop("Haze_thick")

    return list(cloud_parameter_dict.values())


def make_abund(gas_log_abundances: NDArray[np.float_]):
    if len(gas_log_abundances) == 0:
        return np.array([1.0])
        # mmw = 2.28
        # rxsec = 0.0

        return abund, mmw, rxsec

    else:
        abund = np.zeros(g2 - g1 + 1)
        asum = 0
        for i in range(g1, g2):
            abund[i - g1 + 1] = x[i]
            asum = asum + 10 ** x[i]
        # abund = make_gaussian_profile(np.asarray([abund]*number_of_atmospheric_layers).T)
        abund[0] = 1.0 - asum


def make_mmw_and_rxsec():
    mmw, rxsec = af.GetScaOpac(gases, abund[1:])

    return mmw, rxsec


def make_tplong():
    pass
