import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple, Optional

import numpy as np
from numpy.typing import NDArray

from useful_internal_functions import compose

APOLLO_DIRECTORY = Path.cwd().absolute()
if str(APOLLO_DIRECTORY) not in sys.path:
    sys.path.append(str(APOLLO_DIRECTORY))

from apollo.Apollo_ProcessInputs import (  # noqa: E402
    BinIndices,
    SpectrumWithWavelengths,
    calculate_total_flux_in_CGS,
)
from apollo.Apollo_ReadInputsfromFile import FluxScaler, RadiusInputType  # noqa: E402
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


def apply_nothing(argument: Any) -> Any:
    return argument


def calculate_solid_angle(
    radius_case: RadiusInputType, radius_parameter: float, distance_to_system: float
) -> float:
    angle_subtended_by_planet = radius_parameter / (distance_to_system * parsec_in_cm)

    return angle_subtended_by_planet**2


class NormalizationParameters(NamedTuple):
    norad: bool = False
    snormtrunc: Optional[float] = None
    enormtrunc: Optional[float] = None


def normalize_spectrum(
    model_spectrum: SpectrumWithWavelengths,
    normalization_parameters: Optional[NormalizationParameters] = None,
) -> SpectrumWithWavelengths:
    # TODO: untested

    total_flux: float = calculate_total_flux_in_CGS(model_spectrum)

    normspec: NDArray[np.float64] = NormSpec(
        *model_spectrum,
        normalization_parameters.snormtrunc,
        normalization_parameters.enormtrunc,
    )

    normspec: NDArray[np.float64] = (
        normspec * total_flux / np.sum(normspec)
        if normalization_parameters.norad
        else normspec
    )

    return SpectrumWithWavelengths(
        wavelengths=model_spectrum.wavelengths, flux=normspec
    )


class WavelengthCalibrationParameters(NamedTuple):
    model_spectrum_bin_indices: BinIndices
    unit_bin_indices_shift: BinIndices
    wavelength_calibration_parameter: float


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


class BinningParameters(NamedTuple):
    band_index: list[tuple]
    model_spectrum_indices: NDArray[np.int_]
    bin_indices: BinIndices
    binning_factor: float
    convolving_factor: float


def bin_and_convolve_model(
    full_resolution_observed_flux: NDArray[np.float64],
    binned_wavelengths: NDArray[np.float64],
    binning_parameters: BinningParameters,
) -> NDArray[np.float64]:
    lower_bin_indices, upper_bin_indices = binning_parameters.bin_indices

    binw: float = (lower_bin_indices[1] - lower_bin_indices[0]) * (
        binning_parameters.convolving_factor / binning_parameters.binning_factor
    )

    # convmod: list[NDArray[np.float64]] = []
    # for i in range(0, len(modindex)):
    #    convmod.append(ConvSpec(normspec[modindex[i][0] : modindex[i][1]], binw))

    convmod: list[NDArray[np.float64]] = [
        ConvSpec(
            flux=full_resolution_observed_flux[model_index_start:model_index_end],
            bin_width=binw,
        )
        for (
            model_index_start,
            model_index_end,
        ) in binning_parameters.model_spectrum_indices
    ]

    convmod: list[float] = [item for sublist in convmod for item in sublist]

    binmod: list[NDArray[np.float64]] = [
        BinModel(
            flux=convmod,
            binlo=lower_bin_indices[model_start_index : model_end_index + 1],
            binhi=upper_bin_indices[model_start_index : model_end_index + 1],
        )
        for (
            model_start_index,
            model_end_index,
        ) in binning_parameters.model_spectrum_indices
    ]

    return SpectrumWithWavelengths(
        wavelengths=binned_wavelengths,
        flux=np.array([item for sublist in binmod for item in sublist]),
    )


def scale_flux_by_band(
    spectral_quantity: SpectrumWithWavelengths,
    flux_scaler: FluxScaler,
) -> SpectrumWithWavelengths:
    wavelengths = spectral_quantity.wavelengths
    flux = spectral_quantity.flux

    return SpectrumWithWavelengths(
        wavelengths=wavelengths,
        flux=np.where(
            np.logical_and(
                flux_scaler.band_lower_wavelength_boundary <= wavelengths,
                wavelengths < flux_scaler.band_upper_wavelength_boundary,
            ),
            flux * flux_scaler.scale_factor,
            flux,
        ),
    )


class ResolvedAngleScaler(NamedTuple):
    radius_case: RadiusInputType
    radius_parameter: float
    distance_to_system: float


def make_observation_at_full_resolution(
    spectrum_at_system: SpectrumWithWavelengths,
    observation_scaler: Optional[ResolvedAngleScaler] = None,
) -> SpectrumWithWavelengths:
    result = SpectrumWithWavelengths(
        wavelengths=spectrum_at_system.wavelengths,
        flux=spectrum_at_system.flux
        if observation_scaler is None
        else spectrum_at_system.flux * calculate_solid_angle(*observation_scaler),
    )

    return result


def make_observation_at_binned_resolution(
    observation_at_full_resolution: SpectrumWithWavelengths,
    binned_wavelengths: NDArray[np.float64],
    binning_parameters: BinningParameters,
) -> SpectrumWithWavelengths:
    return bin_and_convolve_model(
        full_resolution_observed_flux=observation_at_full_resolution.flux,
        binned_wavelengths=binned_wavelengths,
        binning_parameters=binning_parameters,
    )


def generate_observation_pipeline_from_model_parameters(
    observation_scaler_inputs: Optional[ResolvedAngleScaler] = None,
    flux_scaler_inputs: Optional[list[FluxScaler]] = None,
    binned_wavelengths: Optional[NDArray[np.float64]] = None,
    binning_parameters_inputs: Optional[BinningParameters] = None,
) -> Callable[[SpectrumWithWavelengths], SpectrumWithWavelengths]:
    sequence_of_functions: list[Callable] = [
        make_observation_at_full_resolution
        if observation_scaler_inputs is None
        else partial(
            make_observation_at_full_resolution,
            observation_scaler=observation_scaler_inputs,
        ),
        apply_nothing
        if flux_scaler_inputs is None
        else compose(
            *[
                partial(scale_flux_by_band, flux_scaler=flux_scaler_input)
                for flux_scaler_input in flux_scaler_inputs
            ]
        ),
        apply_nothing
        if (binned_wavelengths is None or binning_parameters_inputs is None)
        else partial(
            make_observation_at_binned_resolution,
            binned_wavelengths=binned_wavelengths,
            binning_parameters=binning_parameters_inputs,
        ),
    ]

    return compose(*sequence_of_functions)
