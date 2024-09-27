import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from xarray_dataclasses import AsDataset, Attr, Coord, Data

APOLLO_DIRECTORY = Path.cwd().absolute()
if str(APOLLO_DIRECTORY) not in sys.path:
    sys.path.append(str(APOLLO_DIRECTORY))

from dataset.builders import prep_unit_registry  # noqa: E402

prep_unit_registry()

wavelength = Literal["wavelength"]


@dataclass
class Wavelength:
    wavelength_bin_starts: float


@dataclass
class Spectrum(AsDataset):
    """Spectral information as Dataset."""

    wavelength_bin_starts: Data[wavelength, float]
    wavelength_bin_ends: Data[wavelength, float]
    flux: Data[wavelength, float]
    wavelengths: Coord[wavelength, float]
    units: Attr[str] = "microns"


spectrum = Spectrum.new(
    wavelength_bin_starts=np.array([0.0, 0.1, 0.2]),
    wavelength_bin_ends=np.array([0.1, 0.2, 0.3]),
    flux=np.array([1.0, 2.0, 3.0]),
    wavelengths=np.array([1.1, 1.2, 1.3]),
)

spectrum = spectrum.pint.quantify({"wavelength_bin_starts": "micrometers"})
print(spectrum)
