from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.interpolate import PchipInterpolator as monotonic_interpolation
from scipy.ndimage import gaussian_filter1d as gaussian_smoothing

from apollo.submodels.function_model import make_model


########## ANJALI PIETTE et. al. Profile ##########
# A specific interpolation with its own smoothing. Should be flexible for
# most retrieval cases without thermal inversions. See for reference
# Piette, Anjali A. A., and Nikku Madhusudhan. “Considerations for Atmospheric
# Retrieval of High-Precision Brown Dwarf Spectra” 19, no. July (July 29, 2020):
# 1–19. http://arxiv.org/abs/2007.15004.
@make_model(
    path_to_metadata=Path.cwd() / "apollo/submodels/TP_models/modified_piette.toml"
)
def piette(
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
    pressures: Sequence[float],  # log(P/bars)
):
    LOGP_NODES = np.array([-4, -3, -2, -1, 0, 0.5, 1, 1.5, 2, 2.5])
    T_NODES = np.array([T_m4, T_m3, T_m2, T_m1, T_0, T_0p5, T_1, T_1p5, T_2, T_2p5])

    interpolated_function = monotonic_interpolation(LOGP_NODES, T_NODES)

    TP_profile = gaussian_smoothing(interpolated_function(pressures), sigma=0.3)
    return TP_profile


###############################################################################
