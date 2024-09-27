import numpy as np
from numpy.typing import NDArray

from apollo.Apollo_ProcessInputs import SpectrumWithWavelengths


def calculate_log_likelihood_with_white_noise_model(
    model_spectrum: SpectrumWithWavelengths,
    data_spectrum: SpectrumWithWavelengths,
    error_in_spectrum: SpectrumWithWavelengths,
    error_inflation_factor: float,
) -> float:
    additional_white_noise_variance: float = np.exp(error_inflation_factor)

    cumulative_variance: NDArray[np.float64] = (
        error_in_spectrum.flux + additional_white_noise_variance
    )

    residuals: NDArray[np.float64] = (
        data_spectrum.flux - model_spectrum.flux
    ) ** 2 / cumulative_variance

    likelihood = -0.5 * np.sum(residuals + np.log(2 * np.pi * cumulative_variance))
    return likelihood


def sample_unit_cube() -> NDArray[np.float64]: ...


def calculate_log_posterior() -> float: ...
