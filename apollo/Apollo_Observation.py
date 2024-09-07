import sys
from os.path import abspath
from typing import NamedTuple, Protocol, TypedDict

import numpy as np
from nptyping import NDArray, Shape

APOLLO_DIRECTORY = abspath(
    "/Users/arthur/Documents/Astronomy/2019/Retrieval/Code/APOLLO"
)
if APOLLO_DIRECTORY not in sys.path:
    sys.path.append(APOLLO_DIRECTORY)

from apollo.Apollo_ProcessInputs import (  # noqa: E402
    BandSpecs,
    BinIndices,
    SpectrumWithWavelengths,
    calculate_total_flux_in_CGS,
)
from apollo.Apollo_ReadInputsfromFile import (  # noqa: E402
    ObservationMode,
    RadiusInputType,
)
from apollo.src.ApolloFunctions import BinModel, ConvSpec, NormSpec  # noqa: E402

REarth_in_cm = 6.371e8
parsec_in_cm = 3.086e18
RJup_in_REarth = 11.2

"""
J_boundaries = [1.10, 1.36]
H_boundaries = [1.44, 1.82]
K_boundaries = [1.94, 2.46]
G395_boundaries = [2.8, 5.3]
G395_ch1_boundaries = [2.8, 4.05]
G395_ch2_boundaries = [4.15, 5.3]
"""


def calculate_solid_angle(
    radius_case: RadiusInputType, radius_parameter: float, distance_to_system: float
) -> float:
    if radius_case == "Rad":
        theta_planet = (radius_parameter * REarth_in_cm) / (
            distance_to_system * parsec_in_cm
        )
    elif radius_case == "RtoD":
        theta_planet = 10**radius_parameter
    elif radius_case == "RtoD2U":
        theta_planet = (
            np.sqrt(radius_parameter)
            * REarth_in_cm
            / (distance_to_system * parsec_in_cm)
        )
    else:
        theta_planet = (RJup_in_REarth * REarth_in_cm) / (
            distance_to_system * parsec_in_cm
        )

    return theta_planet**2


def apply_wavelength_calibration(
    model_spectrum_bin_indices: BinIndices,
    unit_bin_indices_shift: BinIndices,
    wavelength_calibration_parameter: float,
) -> BinIndices:
    ibinlo, ibinhi = model_spectrum_bin_indices
    delibinlo, delibinhi = unit_bin_indices_shift

    # Adjust for wavelength calibration error
    newibinlo = ibinlo + delibinlo * wavelength_calibration_parameter
    newibinhi = ibinhi + delibinhi * wavelength_calibration_parameter

    return BinIndices(lower_bin_index=newibinlo, upper_bin_index=newibinhi)


def normalize_spectrum(
    model_wavelengths: NDArray[np.float_],
    model_incident_flux: NDArray[np.float_],
    norad=False,
    snormtrunc=None,
    enormtrunc=None,
):
    total_flux = calculate_total_flux_in_CGS(...)

    normspec = NormSpec(model_wavelengths, model_incident_flux, snormtrunc, enormtrunc)

    normspec = normspec * total_flux / np.sum(normspec) if norad else normspec

    return normspec


def bin_and_convolve_model(
    full_resolution_observed_model: NDArray[np.float_],
    band_index: list[tuple],
    model_spectrum_indices: NDArray[np.int_],
    bin_indices: BinIndices,
    convolving_factor: float,
    binning_factor: float,
) -> NDArray[np.float_]:
    lower_bin_indices, upper_bin_indices = bin_indices

    binw: float = (lower_bin_indices[1] - lower_bin_indices[0]) * (
        convolving_factor / binning_factor
    )

    # convmod: list[NDArray[np.float_]] = []
    # for i in range(0, len(modindex)):
    #    convmod.append(ConvSpec(normspec[modindex[i][0] : modindex[i][1]], binw))

    convmod: list[NDArray[np.float_]] = [
        ConvSpec(
            flux=full_resolution_observed_model[model_index_start:model_index_end],
            bin_width=binw,
        )
        for model_index_start, model_index_end in zip(
            model_spectrum_indices[:-1], model_spectrum_indices[1:]
        )
    ]
    convmod: list[float] = [item for sublist in convmod for item in sublist]

    binmod: list[NDArray[np.float_]] = [
        BinModel(
            flux=convmod,
            binlo=lower_bin_indices[band_start_index : band_end_index + 1],
            binhi=upper_bin_indices[band_start_index : band_end_index + 1],
        )
        for band_start_index, band_end_index in zip(band_index[:-1], band_index[1:])
    ]

    return np.array([item for sublist in binmod for item in sublist])


# NOTE: How can this be combined with the ParameterValue class?
class FluxScaler(NamedTuple):
    band_lower_wavelength_boundary: float
    band_upper_wavelength_boundary: float
    scale_factor: NDArray[np.float_]


def scale_flux_by_band(
    spectral_quantity: SpectrumWithWavelengths,
    flux_scaler: FluxScaler,
) -> np.ndarray:
    wavelengths = spectral_quantity.wavelengths
    flux = spectral_quantity.flux

    return np.where(
        np.logical_and(
            flux_scaler.band_lower_wavelength_boundary <= wavelengths,
            wavelengths < flux_scaler.band_upper_wavelength_boundary,
        ),
        flux * flux_scaler.scale_factor,
        flux,
    )


class AngleScaler(TypedDict):
    radius_case: RadiusInputType
    radius_parameter: float
    distance_to_system: float


def make_observation_at_full_resolution(
    spectrum_at_system: SpectrumWithWavelengths,
    observing_mode: ObservationMode,
    angle_scaler: AngleScaler,
) -> SpectrumWithWavelengths:
    model_wavelengths, spectral_quantity_at_system = spectrum_at_system

    spectral_quantity_at_observer: NDArray[np.float_] = (
        spectral_quantity_at_system
        if observing_mode == ObservationMode.TRANSIT
        else spectral_quantity_at_system * calculate_solid_angle(**angle_scaler)
    )

    return SpectrumWithWavelengths(
        wavelengths=model_wavelengths,
        flux=spectral_quantity_at_observer,
    )


"""
class BandSpecs(NamedTuple):
    bandindex: list[tuple]
    modindex: NDArray[np.int_]
    modwave: NDArray[np.float_]
"""


def make_observation_at_binned_resolution(
    observation_at_full_resolution: SpectrumWithWavelengths,
    band_index: list[tuple],
    model_spectrum_indices: NDArray[np.int_],
    bin_indices: BinIndices,
    convolving_factor: float,
    binning_factor: float,
) -> SpectrumWithWavelengths:
    band_specs: BandSpecs = BandSpecs(
        bandindex=band_index,
        modindex=model_spectrum_indices,
        modwave=observation_at_full_resolution.wavelengths,
    )

    return bin_and_convolve_model(
        full_resolution_observed_model=observation_at_full_resolution.flux,
        band_specs=band_specs,
        bin_indices=bin_indices,
        convolving_factor=convolving_factor,
        binning_factor=binning_factor,
    )


class SpectrumScaler(Protocol):
    def __call__(
        self, spectrum: NDArray[Shape["original_number_of_wavelengths"]]
    ) -> NDArray[Shape["original_number_of_wavelengths"]]: ...
