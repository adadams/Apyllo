import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

APOLLO_DIRECTORY = Path.cwd().absolute()
if str(APOLLO_DIRECTORY) not in sys.path:
    sys.path.append(str(APOLLO_DIRECTORY))

from apollo.src.wrapPlanet import PyPlanet  # noqa: E402


def get_spectrum(cclass: PyPlanet) -> NDArray[np.float64]:
    return cclass.get_Spectrum()


def get_clear_spectrum(cclass: PyPlanet) -> NDArray[np.float64]:
    return cclass.get_ClearSpectrum()


def get_fractional_cloud_spectrum(
    cclass: PyPlanet, cloud_fraction: float
) -> NDArray[np.float64]:
    assert 0 <= cloud_fraction <= 1, "Cloud fraction must be between 0 and 1"

    return (1 - cloud_fraction) * get_clear_spectrum(
        cclass
    ) + cloud_fraction * get_spectrum(cclass)


def get_effective_temperature(cclass: PyPlanet) -> float:
    return cclass.get_Teff()


def set_observed_spectrum_function() -> Callable: ...
