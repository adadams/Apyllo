from dataclasses import dataclass
from typing import Sequence


@dataclass
class CloudMakePlanetAdapter:
    cloud_model_index: int
    hazetype_index: int
    number_of_streams: int  # maybe?


@dataclass
class GasMakePlanetAdapter:
    gas_species_indices: Sequence[int]
    gas_opacity_directory: str
    opacity_catalog_name: str
    Teff_opacity_catalog_name: str


@dataclass
class TPMakePlanetAdapter:
    TP_model_switch: int


@dataclass
class WavelengthMakePlanetAdapter:
    model_wavelengths: Sequence[float]
    Teff_calculation_wavelengths: Sequence[float]
