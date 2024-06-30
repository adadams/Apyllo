from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable, TypedDict

import numpy as np
from numpy.typing import NDArray
from pandas.compat import pickle_compat
from pint import Unit

# from tqdm import tqdm
from xarray import DataArray, Dataset, apply_ufunc

from apollo.dataset.accessors import (
    extract_dataset_subset_by_parameter_group,
    extract_free_parameters_from_dataset,
)
from apollo.dataset.builders import (
    VariableBlueprint,
    make_dataset_variables_from_dict,
    organize_parameter_data_in_xarray,
)
from apollo.dataset.IO import prep_and_save_dataset
from apollo.formats.custom_types import Pathlike
from apollo.make_forward_model_from_file import (
    evaluate_model_spectrum,
    prep_inputs_for_model,
)
from apollo.retrieval.results.IO import (
    SamplingResults,
    get_parameter_properties_from_defaults,
)
from apollo.retrieval.results.manipulate_results_datasets import calculate_MLE
from user.forward_models.inputs.parse_APOLLO_inputs import parse_APOLLO_input_file


class RunDatasetBlueprint(TypedDict):
    parameter_names: Sequence[str]
    parameter_values: Sequence[float]
    parameter_units: Sequence[Unit | str]
    parameter_default_string_formattings: Sequence[str]
    parameter_group_names: Sequence[str]
    log_likelihoods: Sequence[float]


def make_run_parameter_dataset(run: RunDatasetBlueprint) -> Dataset:
    return Dataset(
        {
            parameter_name: organize_parameter_data_in_xarray(
                name=parameter_name,
                print_name="",
                value=parameter_value,
                unit=parameter_unit,
                coords={"log_likelihood": run.log_likelihoods},
                string_formatter=string_formatter,
                base_group=parameter_group_name,
            )
            for parameter_name, parameter_value, parameter_unit, string_formatter, parameter_group_name in zip(
                run.parameter_names,
                run.parameter_values,
                run.parameter_units,
                run.parameter_default_string_formattings,
                run.parameter_group_names,
            )
        }
    )


def make_run_dataset_from_APOLLO_parameter_file(
    results_parameter_filepath: Path,
    parameter_values: NDArray[np.float_],
    log_likelihoods: Sequence[float],
    **parsing_kwargs,
) -> Dataset:
    with open(results_parameter_filepath, newline="") as retrieved_file:
        parsed_retrieved_file = parse_APOLLO_input_file(
            retrieved_file, **parsing_kwargs
        )

    retrieved_parameter_names = parsed_retrieved_file["parameter_names"]
    retrieved_parameter_group_slices = parsed_retrieved_file["parameter_group_slices"]

    parameter_properties = get_parameter_properties_from_defaults(
        retrieved_parameter_names, retrieved_parameter_group_slices
    )

    return make_run_parameter_dataset(
        **parameter_properties,
        parameter_values=parameter_values,
        log_likelihoods=log_likelihoods,
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

    TP_dataarray: DataArray = (
        apply_ufunc(
            TP_function,
            *TP_variable_list,
            input_core_dims=[[loop_dimension]] * number_of_variables,
            output_core_dims=[["log_pressure", loop_dimension]],
            kwargs={"pressures": log_pressures},
            keep_attrs=True,
        )
        .transpose(loop_dimension, "log_pressure")
        .assign_coords({"log_pressure": log_pressures})
        .rename("temperatures")
        .pint.quantify(output_temperature_unit)
    )
    return TP_dataarray.to_dataset()


def evaluate_model_spectra_from_dataset(
    free_parameter_dataset: Dataset,
    parameter_filepath: Pathlike,
    output_flux_unit: Unit | str = "ergs / second / cm**3",
    loop_dimension: str = "log_likelihood",
) -> Dataset:
    free_variable_list: list[str] = [
        free_parameter_dataset.get(variable) for variable in free_parameter_dataset
    ]
    number_of_variables: int = len(free_variable_list)

    prepped_inputs_and_wavelengths: dict = prep_inputs_for_model(parameter_filepath)
    model_function: Callable = prepped_inputs_and_wavelengths["prepped_inputs"][
        "model_function"
    ]
    observation = prepped_inputs_and_wavelengths["prepped_inputs"]["observation"]
    binned_wavelengths = prepped_inputs_and_wavelengths["binned_wavelengths"]

    def evaluate_model_spectrum_vectorized(
        *sequences_of_parameters: Sequence[Sequence[float]],
    ):
        number_of_wavelength_bins: int = len(binned_wavelengths)
        number_of_model_runs: int = len(sequences_of_parameters[0])
        spectrum_array: NDArray[np.float_] = np.empty(
            (number_of_model_runs, number_of_wavelength_bins), dtype=np.float_
        )

        # for i, model_parameter_set in enumerate(
        #    tqdm(zip(*sequences_of_parameters), total=number_of_model_runs)
        # ):
        for i, model_parameter_set in enumerate(zip(*sequences_of_parameters)):
            model_parameter_set_without_units: list[float] = [
                parameter.magnitude for parameter in model_parameter_set
            ]
            model_spectrum: NDArray[np.float_] = np.asarray(
                evaluate_model_spectrum(
                    model_function=model_function,
                    observation=observation,
                    model_parameters=model_parameter_set_without_units,
                ),
                dtype=np.float_,
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
    results: SamplingResults,
    MLE_parameters_filepath: Pathlike,
    output_filepath_plus_prefix: Pathlike = None,
    output_file_suffix: str = "samples.nc",
    **parsing_kwargs,
) -> Dataset:
    parameter_samples: NDArray[np.float_] = results.samples
    log_likelihoods: NDArray[np.float_] = results.log_likelihoods

    results_dataset: Dataset = make_run_dataset_from_APOLLO_parameter_file(
        MLE_parameters_filepath,
        parameter_values=parameter_samples,
        log_likelihoods=log_likelihoods,
        **parsing_kwargs,
    )

    if output_filepath_plus_prefix:
        results_output_filepath: Path = Path(
            str(output_filepath_plus_prefix) + f".{output_file_suffix}"
        )

        prep_and_save_dataset(results_dataset, path=results_output_filepath)

    return results_dataset


def compile_contributions_into_dataset(
    original_contributions_filepath: Pathlike,
    MLE_parameters_filepath: Pathlike,
    output_filepath_plus_prefix: Pathlike = None,
    output_file_suffix: str = "contributions.nc",
    log_pressures: Sequence[float] = MIDLAYER_LOG_PRESSURES,
) -> Dataset:
    wavelengths: NDArray[np.float_] = prep_inputs_for_model(MLE_parameters_filepath)[
        "binned_wavelengths"
    ]

    with open(original_contributions_filepath, "rb") as pickle_file:
        contributions: dict[str, NDArray[np.float_]] = pickle_compat.load(pickle_file)

    contributions_coordinates: dict[str, VariableBlueprint] = {
        "wavelength": {
            "dims": "wavelength",
            "data": wavelengths,
            "attrs": {"units": "microns"},
        },
        "log_pressure": {
            "dims": "log_pressure",
            "data": log_pressures,
            "attrs": {"units": "log_10(bars)"},
        },
    }

    contributions_dataset: Dataset = Dataset.from_dict(
        {
            "coords": contributions_coordinates,
            "attrs": {"title": "contribution_functions"},
            "dims": (dimension_names := tuple(contributions_coordinates.keys())),
            "data_vars": make_dataset_variables_from_dict(
                contributions, dimension_names
            ),
        }
    ).transpose("log_pressure", "wavelength")

    if output_filepath_plus_prefix:
        contributions_output_filepath: Path = Path(
            str(output_filepath_plus_prefix) + f".{output_file_suffix}"
        )

        prep_and_save_dataset(contributions_dataset, path=contributions_output_filepath)

    return contributions_dataset


def prepare_MLE_dataset_from_results_dataset(
    results_dataset: Dataset,
    output_filepath_plus_prefix: Pathlike = None,
    output_file_suffix: str = "MLE-parameters.nc",
) -> Dataset:
    MLE_parameters_dataset: Dataset = calculate_MLE(results_dataset)

    if output_filepath_plus_prefix:
        MLE_output_filepath: Path = Path(
            str(output_filepath_plus_prefix) + f".{output_file_suffix}"
        )

        prep_and_save_dataset(MLE_parameters_dataset, path=MLE_output_filepath)

    return MLE_parameters_dataset


def prepare_TP_profile_dataset_from_results_dataset(
    results_dataset: Dataset,
    TP_function: Callable[[Any], Sequence[float]],
    output_filepath_plus_prefix: Pathlike = None,
    output_file_suffix: str = "TP-profiles.nc",
    log_pressures: Sequence[float] = MIDLAYER_LOG_PRESSURES,
) -> Dataset:
    TP_dataset: Dataset = extract_dataset_subset_by_parameter_group(
        results_dataset, group_name="Atm"
    )

    TP_profile_dataset: Dataset = evaluate_TP_functions_from_dataset(
        TP_dataset, TP_function, log_pressures
    )

    if output_filepath_plus_prefix:
        TP_output_filepath: Path = Path(
            str(output_filepath_plus_prefix) + f".{output_file_suffix}"
        )

        prep_and_save_dataset(TP_profile_dataset, path=TP_output_filepath)

    return TP_dataset


def prepare_model_spectra_dataset_from_free_parameters_dataset(
    results_dataset: Dataset,
    MLE_parameters_filepath: Pathlike,
    number_of_resampled_model_spectra: int,
    output_filepath_plus_prefix: Pathlike = None,
    output_file_suffix: str = None,
) -> Dataset:
    free_parameter_dataset: Dataset = extract_free_parameters_from_dataset(
        results_dataset
    )

    model_spectrum_dataset: Dataset = evaluate_model_spectra_from_dataset(
        free_parameter_dataset.tail(log_likelihood=number_of_resampled_model_spectra),
        MLE_parameters_filepath,
    )

    if output_filepath_plus_prefix:
        if output_file_suffix is None:
            output_file_suffix: str = (
                f".last-{number_of_resampled_model_spectra}-spectra.nc"
            )

        model_spectra_filename: Path = Path(
            str(output_filepath_plus_prefix) + f".{output_file_suffix}"
        )

        prep_and_save_dataset(model_spectrum_dataset, path=model_spectra_filename)

    return model_spectrum_dataset
