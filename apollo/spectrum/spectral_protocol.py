from abc import abstractmethod
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


@dataclass
class Spectral(Protocol):
    """Has wavelengths, possibly in binned, possibly in unbinned input format.
    Also has a corresponding spectrum that can be convolved and down-sampled in
    resolution."""

    wavelength_bin_starts: NDArray[np.float_]
    wavelength_bin_ends: NDArray[np.float_]
    wavelengths: NDArray[np.float_]
    spectrum: NDArray[np.float_]

    @abstractmethod
    def convolve(self, *args, **kwargs) -> NDArray[np.float_]:
        raise NotImplementedError

    @abstractmethod
    def down_bin(self, *args, **kwargs) -> NDArray[np.float_]:
        raise NotImplementedError

    @abstractmethod
    def down_resolve(self, *args, **kwargs) -> NDArray[np.float_]:
        raise NotImplementedError
