from dataclasses import dataclass

from numpy.typing import NDArray


@dataclass
class CloudParametersAdapter:
    pass


@dataclass
class ParamsSetParamsAdapter:
    radius: float
    log_gravity: float
    cloud_base_log_pressure: float
    stellar_temperature: float
    stellar_radius: float
    mean_molecular_weight: float
    total_gas_cross_section: float
    minimum_log_pressure: float
    maximum_log_pressure: float
    semimajor_axis: float
    cloud_parameters: CloudParametersAdapter


@dataclass
class OtherSetParamsAdapter:
    gas_log_abundances: NDArray
    gas_cross_sections: NDArray
    TP_profile: NDArray
