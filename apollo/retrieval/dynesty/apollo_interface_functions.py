from numpy.typing import ArrayLike
from pathlib import Path
from typing import Any, Callable, Sequence
from xarray import Dataset

from apollo.general_protocols import Pathlike
from apollo.make_forward_model_from_file import prep_inputs_for_model
from apollo.retrieval.dynesty.parse_dynesty_outputs import (
    load_and_filter_all_parameters_by_importance,
)
from apollo.retrieval.dynesty.build_and_manipulate_datasets import (
    calculate_MLE,
    change_parameter_values_using_MLE_dataset,
    make_run_parameter_dataset,
)
from apollo.retrieval.dynesty.convenience_functions import (
    get_parameter_properties_from_defaults,
)
from apollo import TP_functions

from user.models.inputs.parse_APOLLO_inputs import (
    parse_APOLLO_input_file,
    change_properties_of_parameters,
    write_parsed_input_to_output,
)


def prep_inputs_and_get_binned_wavelengths(
    parameter_filepath: Pathlike,
) -> dict[str, Any]:
    prepped_inputs, binned_wavelengths = prep_inputs_for_model(parameter_filepath)

    return dict(prepped_inputs=prepped_inputs, binned_wavelengths=binned_wavelengths)


def make_MLE_parameter_file_from_input_parameter_file(
    fitting_results_filepath: Pathlike,
    derived_fit_parameters_filepath: Pathlike,
    input_parameters_filepath: Pathlike,
) -> None:
    results = load_and_filter_all_parameters_by_importance(
        fitting_results_filepath, derived_fit_parameters_filepath
    )
    parameter_samples = results["samples"]
    log_likelihoods = results["log_likelihoods"]

    with open(input_parameters_filepath, newline="") as input_file:
        parsed_input_file = parse_APOLLO_input_file(input_file, delimiter=" ")

    input_parameters = parsed_input_file["parameters"]
    input_parameter_names = parsed_input_file["parameter_names"]
    input_parameter_group_slices = parsed_input_file["parameter_group_slices"]
    input_file_headers = parsed_input_file["header"]

    parameter_properties = get_parameter_properties_from_defaults(
        input_parameter_names, input_parameter_group_slices
    )

    sample_dataset = make_run_parameter_dataset(
        **parameter_properties,
        parameter_values=parameter_samples.T,
        log_likelihoods=log_likelihoods,
    )
    MLE_parameters = calculate_MLE(sample_dataset)

    MLE_output_parameter_dict = change_properties_of_parameters(
        input_parameters,
        change_parameter_values_using_MLE_dataset,
        MLE_parameters,
    )

    output_MLE_filename = input_parameters_filepath.replace("input", "retrieved")
    with open(output_MLE_filename, "w", newline="") as output_file:
        write_parsed_input_to_output(
            input_file_headers, MLE_output_parameter_dict, output_file
        )

    return None


def make_dataset_from_APOLLO_parameter_file(
    results_parameter_filepath: Path,
    parameter_values: ArrayLike,
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


def get_TP_function_from_APOLLO_parameter_file(
    parameter_filepath: Pathlike, **parsing_kwargs
) -> Callable[[Any], Sequence[float]]:
    with open(parameter_filepath, newline="") as retrieved_file:
        parsed_retrieved_file = parse_APOLLO_input_file(
            retrieved_file, **parsing_kwargs
        )

    TP_function_name = parsed_retrieved_file["parameters"]["Atm"]["options"][0]

    return getattr(TP_functions, TP_function_name)
