from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from pandas.compat import pickle_compat
from pint import Unit
from tqdm import tqdm
from xarray import Dataset, apply_ufunc

from apollo.convenience_types import Pathlike
from apollo.dataset.dataset_functions import (
    extract_dataset_subset_by_parameter_group,
    extract_free_parameters_from_dataset,
    make_dataset_variables_from_dict,
    save_dataset_with_units,
)
from apollo.make_forward_model_from_file import evaluate_model_spectrum
from apollo.retrieval.dynesty.apollo_interface_functions import (
    make_dataset_from_APOLLO_parameter_file,
    prep_inputs_and_get_binned_wavelengths,
)
from apollo.retrieval.dynesty.build_and_manipulate_datasets import calculate_MLE
from apollo.retrieval.dynesty.parse_dynesty_outputs import (
    load_and_filter_all_parameters_by_importance,
)

ORIGINAL_LOG_PRESSURES: Sequence[float] = np.linspace(-4, 2.5, num=71)
MIDLAYER_LOG_PRESSURES: Sequence[float] = (
    ORIGINAL_LOG_PRESSURES[:-1] + ORIGINAL_LOG_PRESSURES[1:]
) / 2


def evaluate_TP_functions_from_dataset(
    TP_dataset: Dataset,
    TP_function: Callable[[Any, Sequence[float]], Sequence[float]],
    log_pressures: Sequence[float],
    output_temperature_unit: Unit | str = "kelvin",
    loop_dimension: str = "log_likelihood",
) -> Dataset:
    TP_variable_list = [TP_dataset.get(variable) for variable in TP_dataset]
    number_of_variables = len(TP_variable_list)

    return (
        apply_ufunc(
            TP_function,
            *TP_variable_list,
            input_core_dims=[[loop_dimension]] * number_of_variables,
            output_core_dims=[["log_pressure", loop_dimension]],
            kwargs=dict(pressures=log_pressures),
            keep_attrs=True,
        )
        .transpose(loop_dimension, "log_pressure")
        .assign_coords(dict(log_pressure=log_pressures))
        .rename("temperatures")
        .pint.quantify(output_temperature_unit)
    )


def evaluate_model_spectra_from_dataset(
    free_parameter_dataset: Dataset,
    parameter_filepath: Pathlike,
    output_flux_unit: Unit | str = "ergs / second / cm**3",
    loop_dimension: str = "log_likelihood",
) -> Dataset:
    free_variable_list = [
        free_parameter_dataset.get(variable) for variable in free_parameter_dataset
    ]
    number_of_variables = len(free_variable_list)

    prepped_inputs_and_wavelengths = prep_inputs_and_get_binned_wavelengths(
        parameter_filepath
    )
    model_function = prepped_inputs_and_wavelengths["prepped_inputs"]["model_function"]
    observation = prepped_inputs_and_wavelengths["prepped_inputs"]["observation"]
    binned_wavelengths = prepped_inputs_and_wavelengths["binned_wavelengths"]

    def evaluate_model_spectrum_vectorized(
        *sequences_of_parameters: Sequence[Sequence[float]],
    ):
        number_of_wavelength_bins = len(binned_wavelengths)
        number_of_model_runs = len(sequences_of_parameters[0])
        spectrum_array = np.empty(
            (number_of_model_runs, number_of_wavelength_bins), dtype=np.float64
        )

        for i, model_parameter_set in enumerate(
            tqdm(zip(*sequences_of_parameters), total=number_of_model_runs)
        ):
            model_parameter_set_without_units = [
                parameter.magnitude for parameter in model_parameter_set
            ]
            model_spectrum = np.asarray(
                evaluate_model_spectrum(
                    model_function=model_function,
                    observation=observation,
                    model_parameters=model_parameter_set_without_units,
                ),
                dtype=np.float64,
            )
            spectrum_array[i] = model_spectrum

        return spectrum_array

    return (
        apply_ufunc(
            evaluate_model_spectrum_vectorized,
            *free_variable_list,
            input_core_dims=[[loop_dimension]] * number_of_variables,
            output_core_dims=[[loop_dimension, "wavelength"]],
            keep_attrs=True,
        )
        .assign_coords(dict(wavelength=binned_wavelengths))
        .rename("flux")
        .pint.quantify(output_flux_unit)
    )


def compile_results_into_dataset(
    fitting_results_filepath: Pathlike,
    derived_fit_parameters_filepath: Pathlike,
    MLE_parameters_filepath: Pathlike,
    output_filepath_plus_prefix: Pathlike = None,
    output_file_suffix: str = "samples.nc",
    **parsing_kwargs,
) -> Dataset:
    results = load_and_filter_all_parameters_by_importance(
        fitting_results_filepath, derived_fit_parameters_filepath
    )
    parameter_samples = results["samples"]
    log_likelihoods = results["log_likelihoods"]

    results_dataset = make_dataset_from_APOLLO_parameter_file(
        MLE_parameters_filepath,
        parameter_values=parameter_samples,
        log_likelihoods=log_likelihoods,
        **parsing_kwargs,
    )

    if output_filepath_plus_prefix:
        results_output_filepath = Path(
            str(output_filepath_plus_prefix) + f".{output_file_suffix}"
        )

        save_dataset_with_units(results_dataset, path=results_output_filepath)

    return results_dataset


def compile_contributions_into_dataset(
    original_contributions_filepath: Pathlike,
    MLE_parameters_filepath: Pathlike,
    output_filepath_plus_prefix: Pathlike = None,
    output_file_suffix: str = "contributions.nc",
    log_pressures: Sequence[float] = MIDLAYER_LOG_PRESSURES,
) -> Dataset:
    wavelengths = prep_inputs_and_get_binned_wavelengths(MLE_parameters_filepath)[
        "binned_wavelengths"
    ]

    with open(original_contributions_filepath, "rb") as pickle_file:
        contributions = pickle_compat.load(pickle_file)

    contributions_coordinates = dict(
        wavelength=dict(
            dims="wavelength", data=wavelengths, attrs=dict(units="microns")
        ),
        log_pressure=dict(
            dims="log_pressure",
            data=log_pressures,
            attrs=dict(units="log_10(bars)"),
        ),
    )

    contributions_dataset = Dataset.from_dict(
        dict(
            coords=contributions_coordinates,
            attrs=dict(title="contribution_functions"),
            dims=(dimension_names := tuple(contributions_coordinates.keys())),
            data_vars=make_dataset_variables_from_dict(contributions, dimension_names),
        )
    ).transpose("log_pressure", "wavelength")

    if output_filepath_plus_prefix:
        contributions_output_filepath = Path(
            str(output_filepath_plus_prefix) + f".{output_file_suffix}"
        )

        save_dataset_with_units(
            contributions_dataset, path=contributions_output_filepath
        )

    return contributions_dataset


def prepare_MLE_dataset_from_results_dataset(
    results_dataset: Dataset,
    output_filepath_plus_prefix: Pathlike = None,
    output_file_suffix: str = "MLE-parameters.nc",
) -> Dataset:
    MLE_parameters_dataset = calculate_MLE(results_dataset)

    if output_filepath_plus_prefix:
        MLE_output_filepath = Path(
            str(output_filepath_plus_prefix) + f".{output_file_suffix}"
        )

        save_dataset_with_units(MLE_parameters_dataset, path=MLE_output_filepath)

    return MLE_parameters_dataset


def prepare_TP_profile_dataset_from_results_dataset(
    results_dataset: Dataset,
    TP_function: Callable[[Any], Sequence[float]],
    output_filepath_plus_prefix: Pathlike = None,
    output_file_suffix: str = "TP-profiles.nc",
    log_pressures: Sequence[float] = MIDLAYER_LOG_PRESSURES,
) -> Dataset:
    TP_dataset = extract_dataset_subset_by_parameter_group(
        results_dataset, group_name="Atm"
    )

    TP_profile_dataset = evaluate_TP_functions_from_dataset(
        TP_dataset, TP_function, log_pressures
    )

    if output_filepath_plus_prefix:
        TP_output_filepath = Path(
            str(output_filepath_plus_prefix) + f".{output_file_suffix}"
        )

        save_dataset_with_units(TP_profile_dataset, path=TP_output_filepath)

    return TP_dataset


def prepare_model_spectra_dataset_from_free_parameters_dataset(
    results_dataset: Dataset,
    MLE_parameters_filepath: Pathlike,
    output_filepath_plus_prefix: Pathlike = None,
    number_of_resampled_model_spectra: int = 100,
    output_file_suffix: str = None,
) -> Dataset:
    free_parameter_dataset = extract_free_parameters_from_dataset(results_dataset)

    model_spectrum_dataset = evaluate_model_spectra_from_dataset(
        free_parameter_dataset.tail(log_likelihood=number_of_resampled_model_spectra),
        MLE_parameters_filepath,
    )

    if output_filepath_plus_prefix:
        if output_file_suffix is None:
            output_file_suffix = f".last-{number_of_resampled_model_spectra}-spectra.nc"

        model_spectra_filename = Path(
            str(output_filepath_plus_prefix) + f".{output_file_suffix}"
        )

        save_dataset_with_units(model_spectrum_dataset, path=model_spectra_filename)

    return model_spectrum_dataset
