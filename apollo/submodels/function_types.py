from typing import Protocol, TypeVar

import numpy as np
from nptyping import Float, NDArray, Shape, Structure

VerticalResolution = TypeVar("VerticalResolution", bound=int)
SpectralResolution = TypeVar("SpectralResolution", bound=int)
AngularResolution = TypeVar("AngularResolution", bound=int)

SpectralArray = NDArray[Shape[SpectralResolution], Structure[{"wavelength": Float}]]
AngleArray = NDArray[Shape[AngularResolution], Structure[{"angle": Float}]]
PressureArray = NDArray[Shape[VerticalResolution], Structure[{"pressure": Float}]]
TemperatureArray = NDArray[Shape[VerticalResolution], Structure[{"temperature": Float}]]


class MaterialFunction(Protocol):
    def __call__(
        self,
        *args,
        wavelengths: SpectralArray,
        log_pressures: PressureArray,
        scattering_angles: AngleArray,
        **kwargs,
    ) -> NDArray[np.float64]: ...


class TPFunction(Protocol):
    def __call__(
        self,
        *args,
        log_pressures: PressureArray,
        **kwargs,
    ) -> TemperatureArray: ...
