import sys
from os.path import abspath
from typing import Sequence

import numpy as np
from xarray import DataArray, Dataset

APOLLO_DIRECTORY = abspath(
    "/Users/arthur/Documents/Astronomy/2019/Retrieval/Code/APOLLO"
)
sys.path.append(APOLLO_DIRECTORY)

from apollo.data.read_into_xarray import read_APOLLO_data_into_dataset
from apollo.dataset_functions import load_dataset_with_units
from apollo.general_protocols import Pathlike
from apollo.retrieval.dynesty.parse_dynesty_outputs import unpack_results_filepaths
from user.results.process_dynesty_results import RESULTS_DIRECTORY


def calculate_likelihood_error_term(
    error_terms: Sequence[float], added_white_noise_variance: float
) -> float:
    inflated_error_terms = np.log(
        2 * np.pi * (error_terms**2 + added_white_noise_variance)
    )

    return np.sum(inflated_error_terms)


def calculate_reduced_chi_squared_from_log_likelihood(
    error_terms: Sequence[float],
    added_white_noise_variance: float,
    maximum_log_likelihood: float,
    number_of_parameters: int,
) -> float:
    number_of_data_points = len(error_terms)

    likelihood_error_term = calculate_likelihood_error_term(
        error_terms, added_white_noise_variance
    )

    reduced_chi_squared = -(2 * maximum_log_likelihood + likelihood_error_term) / (
        number_of_data_points - number_of_parameters
    )
    return reduced_chi_squared


def calculate_reduced_chi_squared_directly(
    MLE_spectrum: Sequence[float],
    data: Sequence[float],
    error_terms: Sequence[float],
    added_white_noise_variance: float,
    number_of_parameters: int,
) -> float:
    number_of_data_points = len(MLE_spectrum)

    inflated_variance = error_terms**2 + added_white_noise_variance

    standard_variance = (MLE_spectrum - data) ** 2 / inflated_variance

    reduced_chi_squared = np.sum(standard_variance) / (
        number_of_data_points - number_of_parameters
    )
    return reduced_chi_squared


def calculate_reduced_chi_square_for_runs(
    results_filepath_directory: Pathlike, inflate_errors: bool = True
) -> list[float]:
    results_filepath_dictionary = unpack_results_filepaths(results_filepath_directory)

    reduced_chi_squared = {}
    direct_reduced_chi_squared = {}
    for run_name, results_filepaths in results_filepath_dictionary.items():
        data_filepath = results_filepaths["data"]
        MLE_spectrum_filepath = results_filepaths["MLE_model_spectrum"]
        samples_dataset_filepath = results_filepaths["samples_dataset"]

        data: Dataset = read_APOLLO_data_into_dataset(data_filepath)
        error_terms: DataArray = data.lower_errors.pint.dequantify()
        _, _, model_spectrum, _, _, _ = np.loadtxt(MLE_spectrum_filepath).T

        samples_dataset: Dataset = load_dataset_with_units(samples_dataset_filepath)

        maximum_log_likelihood: float = samples_dataset.log_likelihood.max().item()

        added_white_noise_variance: float = (
            np.exp(
                samples_dataset.logf.isel(
                    samples_dataset.log_likelihood.argmax(...)
                ).pint.dequantify()
            ).item()
            if inflate_errors
            else 0
        )

        number_of_parameters = len(samples_dataset.data_vars)

        reduced_chi_squared[run_name] = (
            calculate_reduced_chi_squared_from_log_likelihood(
                error_terms,
                added_white_noise_variance,
                maximum_log_likelihood,
                number_of_parameters,
            ).item()
        )

        direct_reduced_chi_squared[run_name] = calculate_reduced_chi_squared_directly(
            model_spectrum,
            data.spectrum.pint.dequantify(),
            error_terms,
            added_white_noise_variance,
            number_of_parameters,
        ).item()

        added_noise_ratio = calculate_ratio_of_added_noise_to_original_RMS_noise(
            error_terms, added_white_noise_variance
        ).item()
        print(f"{run_name}: {added_noise_ratio=}")

    return reduced_chi_squared, direct_reduced_chi_squared


def calculate_ratio_of_added_noise_to_original_RMS_noise(
    error_terms: Sequence[float], added_white_noise_variance: float
) -> float:
    original_mean_variance: float = np.sum(error_terms**2) / len(error_terms)

    return np.sqrt(added_white_noise_variance / original_mean_variance)


if __name__ == "__main__":
    OBJECT_NAME = "2M2236"

    SPECIFIC_RESULTS_DIRECTORY = RESULTS_DIRECTORY / OBJECT_NAME

    reduced_chi_squared, direct_reduced_chi_squared = (
        calculate_reduced_chi_square_for_runs(
            SPECIFIC_RESULTS_DIRECTORY, inflate_errors=False
        )
    )

    print(f"{reduced_chi_squared=}")
