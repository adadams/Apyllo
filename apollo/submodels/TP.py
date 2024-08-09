from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import PchipInterpolator as monotonic_interpolation
from scipy.ndimage import gaussian_filter1d as gaussian_smoothing

from apollo.submodels.function_model import make_model


@runtime_checkable
class isTPFunction(Protocol):
    def __call__(
        self, *args, log_pressures: ArrayLike, **kwargs
    ) -> NDArray[np.float_]: ...


########## ANJALI PIETTE et. al. Profile ##########
# A specific interpolation with its own smoothing. Should be flexible for
# most retrieval cases without thermal inversions. See for reference
# Piette, Anjali A. A., and Nikku Madhusudhan. “Considerations for Atmospheric
# Retrieval of High-Precision Brown Dwarf Spectra” 19, no. July (July 29, 2020):
# 1–19. http://arxiv.org/abs/2007.15004.
def general_piette_function(
    *temperatures: Sequence[float],
    log_pressure_nodes: NDArray[np.float_],
    log_pressures: NDArray[np.float_],
    smoothing_parameter: float,
):
    interpolated_function = monotonic_interpolation(log_pressure_nodes, temperatures)

    TP_profile = gaussian_smoothing(
        interpolated_function(log_pressures), sigma=smoothing_parameter
    )
    return TP_profile


modified_piette_metadata = """
dimensional_arguments = ['pressures']

[parameters]
T_m4 =  { print_name = '$T_{-4}$', units = 'kelvin' }
T_m3 =  { print_name = '$T_{-3}$', units = 'kelvin' }
T_m2 =  { print_name = '$T_{-2}$', units = 'kelvin' }
T_m1 =  { print_name = '$T_{-1}$', units = 'kelvin' }
T_0 =   { print_name = '$T_{0}$', units = 'kelvin' }
T_0p5 = { print_name = '$T_{0.5}$', units = 'kelvin' }
T_1 =   { print_name = '$T_{1}$', units = 'kelvin' }
T_1p5 = { print_name = '$T_{1.5}$', units = 'kelvin' }
T_2 =   { print_name = '$T_{2}$', units = 'kelvin' }
T_2p5 = { print_name = '$T_{2.5}$', units = 'kelvin' }
"""


@make_model(path_to_metadata=modified_piette_metadata)
def modified_piette(
    T_m4: float,
    T_m3: float,
    T_m2: float,
    T_m1: float,
    T_0: float,
    T_0p5: float,
    T_1: float,
    T_1p5: float,
    T_2: float,
    T_2p5: float,
    pressures: NDArray[np.float_],
) -> NDArray[np.float_]:
    log_pressure_nodes = np.array([-4, -3, -2, -1, 0, 0.5, 1, 1.5, 2, 2.5])

    return general_piette_function(
        T_m4,
        T_m3,
        T_m2,
        T_m1,
        T_0,
        T_0p5,
        T_1,
        T_1p5,
        T_2,
        T_2p5,
        log_pressure_nodes=log_pressure_nodes,
        log_pressures=pressures,
        smoothing_parameter=0.3,
    )


###############################################################################
