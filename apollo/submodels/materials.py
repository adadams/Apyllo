from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import PchipInterpolator as monotonic_interpolation
from scipy.ndimage import gaussian_filter1d as gaussian_smoothing

from apollo.submodels.function_model import make_model


@runtime_checkable
class isMaterialFunction(Protocol):
    def __call__(
        self, *args, log_pressures: ArrayLike, **kwargs
    ) -> NDArray[np.float_]: ...
