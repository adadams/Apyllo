from abc import ABC
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


class CloudParametersAdapter(ABC):
    def make_list_of_cloud_parameters(self) -> list[float]:
        pass

    def get_cloud_model_index(self) -> int:
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
    gas_log_abundances: NDArray[np.float_]
    gas_cross_sections: NDArray[np.float_]
    TP_profile: NDArray[np.float_]
