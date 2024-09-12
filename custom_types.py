from pathlib import Path
from typing import IO

type Pathlike = Path | str
type Filelike = Pathlike | IO

"""
class SpectrumEndofunction(Protocol):
    def __call__(
        self, spectrum: NDArray[Shape["original_number_of_wavelengths"]]
    ) -> NDArray[Shape["original_number_of_wavelengths"]]: ...
"""
