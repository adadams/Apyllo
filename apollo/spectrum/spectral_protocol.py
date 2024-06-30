from typing import Protocol

from numpy.typing import ArrayLike


class Spectral(Protocol):
    """Has wavelengths, possibly in binned, possibly in unbinned input format.
    Also has a corresponding spectrum that can be convolved and down-sampled in
    resolution."""

    wavelength_bin_starts: ArrayLike
    wavelength_bin_ends: ArrayLike
    wavelengths: ArrayLike
    flux: ArrayLike

    def convolve(self, *args, **kwargs) -> ArrayLike: ...

    def down_bin(self, *args, **kwargs) -> ArrayLike: ...

    def down_resolve(self, *args, **kwargs) -> ArrayLike: ...
