import sys
from os.path import abspath

import numpy as np
from numpy.typing import NDArray

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
    radius_case: RadiusInputType, radius_parameter, dist
) -> float:
    if radius_case == "Rad":
        theta_planet = (radius_parameter * REarth_in_cm) / (dist * parsec_in_cm)
    elif radius_case == "RtoD":
        theta_planet = 10**radius_parameter
    elif radius_case == "RtoD2U":
        theta_planet = np.sqrt(radius_parameter) * REarth_in_cm / (dist * parsec_in_cm)
    else:
        theta_planet = (RJup_in_REarth * REarth_in_cm) / (dist * parsec_in_cm)

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
    full_resolution_observed_model: BandSpecs,  # i.e. substitute normspec for modflux
    bin_indices: BinIndices,
    convolving_factor: float,
    binning_factor: float,
) -> NDArray[np.float_]:
    normspec, modindex, bandindex = full_resolution_observed_model

    lower_bin_index, upper_bin_index = bin_indices

    binw: float = (lower_bin_index[1] - lower_bin_index[0]) * (
        convolving_factor / binning_factor
    )

    convmod: list[NDArray[np.float_]] = []
    for i in range(0, len(modindex)):
        convmod.append(ConvSpec(normspec[modindex[i][0] : modindex[i][1]], binw))
    convmod = [item for sublist in convmod for item in sublist]
    # convmod = af.ConvSpec(fincident,binw)
    # convmod = fincident

    binmod_list: list[NDArray[np.float_]] = []
    for i in range(0, len(modindex)):
        binmod_piece = BinModel(
            convmod,
            lower_bin_index[bandindex[i][0] : (bandindex[i][1] + 1)],
            upper_bin_index[bandindex[i][0] : (bandindex[i][1] + 1)],
        )
        binmod_list.append(binmod_piece)

    binmod = [item for sublist in binmod_list for item in sublist]

    return binmod


def scale_flux_by_band(
    flux: np.ndarray,
    wavelengths: np.ndarray,
    band_lower_wavelength_boundary: float,
    band_upper_wavelength_boundary: float,
    scale_factor: float,
) -> np.ndarray:
    return np.where(
        np.logical_and(
            band_lower_wavelength_boundary <= wavelengths,
            wavelengths <= band_upper_wavelength_boundary,
        ),
        flux * scale_factor,
        flux,
    )


def make_observation_at_full_resolution(
    model_wavelengths: NDArray[np.float_],
    observing_mode: ObservationMode,
    spectral_quantity_at_system: NDArray[np.float_],
    radius_case: RadiusInputType,
    radius_parameter: float,
    distance_to_system: float,
) -> SpectrumWithWavelengths:
    spectral_quantity_at_observer: NDArray[np.float_] = (
        spectral_quantity_at_system
        if observing_mode == ObservationMode.TRANSIT
        else spectral_quantity_at_system
        * calculate_solid_angle(radius_case, radius_parameter, dist=distance_to_system)
    )


def make_observation_at_binned_resolution() -> SpectrumWithWavelengths: ...
