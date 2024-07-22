"""
class CloudModel(StrEnum):
    no_clouds = auto()
    opaque_deck = auto()
    single_particle_size_uniform_number_density = auto()
    single_particle_size_gaussian_number_density = auto()
    power_law_opacity = auto()
"""

from typing import NamedTuple, TypedDict


class SingleParticleSizeUniformNumberDensityClouds(NamedTuple):
    cloud_number_density: float
    cloud_particle_size: float
    cloud_minimum_log_pressure: float
    cloud_maximum_log_pressure: (
        float  # NOTE: NOT the input parameter, which is thickness.
    )


class SingleParticleSizeGaussianNumberDensityClouds(NamedTuple):
    cloud_peak_number_density: float
    cloud_particle_size: float
    cloud_peak_log_pressure: float
    cloud_log_pressure_scale: float


class PowerLawOpacityClouds(NamedTuple):
    opacity_power_law_exponent: float
    cloud_minimum_log_pressure: float
    cloud_log_pressure_thickness: float
    cloud_reference_log_optical_depth: float
    cloud_single_scattering_albedo: float


class PatchyCloudParameters(TypedDict):
    cloud_parameters: NamedTuple
    cloud_cover_fraction: float = 1.0
