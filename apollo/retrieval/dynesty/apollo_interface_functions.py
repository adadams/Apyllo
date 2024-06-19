import sys
from operator import itemgetter
from os.path import abspath
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import tomllib
from numpy.typing import NDArray
from xarray import Dataset

APOLLO_DIRECTORY = abspath(
    "/Users/arthur/Documents/Astronomy/2019/Retrieval/Code/APOLLO"
)
if APOLLO_DIRECTORY not in sys.path:
    sys.path.append(APOLLO_DIRECTORY)

from apollo.convenience_types import Pathlike  # noqa: E402
from apollo.make_forward_model_from_file import prep_inputs_for_model  # noqa: E402
from apollo.retrieval.dynesty.build_and_manipulate_datasets import (  # noqa: E402
    calculate_MLE,
    change_parameter_values_using_MLE_dataset,
    make_run_parameter_dataset,
)
from apollo.retrieval.dynesty.convenience_functions import (  # noqa: E402
    get_parameter_properties_from_defaults,
)
from apollo.retrieval.dynesty.parse_dynesty_outputs import (  # noqa: E402
    load_and_filter_all_parameters_by_importance,
)
from apollo.submodels import TP  # noqa: E402
from user.forward_models.inputs.parse_APOLLO_inputs import (  # noqa: E402
    change_properties_of_parameters,
    parse_APOLLO_input_file,
    write_parsed_input_to_output,
)


def get_print_names_from_parameter_names(
    parameter_names: Sequence[str], parameter_spec_filepath: Pathlike
) -> Sequence[str]:
    with open(parameter_spec_filepath, "rb") as parameter_spec_file:
        parameter_specs = tomllib.load(parameter_spec_file)

    name_getter = itemgetter(*parameter_names)

    return [
        parameter_spec["print_name"] for parameter_spec in name_getter(parameter_specs)
    ]


def prep_inputs_and_get_binned_wavelengths(
    parameter_filepath: Pathlike,
) -> dict[str, Any]:
    prepped_inputs, binned_wavelengths = prep_inputs_for_model(parameter_filepath)

    return dict(prepped_inputs=prepped_inputs, binned_wavelengths=binned_wavelengths)


def create_MLE_output_dictionary(
    fitting_results_filepath: Pathlike,
    derived_fit_parameters_filepath: Pathlike,
    input_parameters_filepath: Pathlike,
) -> dict[str, Any]:
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

    parameter_properties = get_parameter_properties_from_defaults(
        input_parameter_names, input_parameter_group_slices
    )
    parameter_print_names = get_print_names_from_parameter_names(
        input_parameter_names, "user/results/2M2236/parameter_print_attributes.toml"
    )

    sample_dataset = make_run_parameter_dataset(
        **parameter_properties,
        parameter_print_names=parameter_print_names,
        parameter_values=parameter_samples,
        log_likelihoods=log_likelihoods,
    )
    MLE_parameters = calculate_MLE(sample_dataset)

    MLE_output_parameter_dict = change_properties_of_parameters(
        input_parameters,
        change_parameter_values_using_MLE_dataset,
        MLE_parameters,
    )

    MLE_output_dict = {
        **{
            name: value
            for name, value in parsed_input_file.items()
            if name != "parameters"
        },
        "parameters": MLE_output_parameter_dict,
    }

    return MLE_output_dict


def make_MLE_parameter_file_from_input_parameter_file(
    fitting_results_filepath: Pathlike,
    derived_fit_parameters_filepath: Pathlike,
    input_parameters_filepath: Pathlike,
    data_filepath: Pathlike,
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

    input_file_headers["Data"] = (str(data_filepath.relative_to(Path.cwd())),)

    parameter_properties = get_parameter_properties_from_defaults(
        input_parameter_names, input_parameter_group_slices
    )

    sample_dataset = make_run_parameter_dataset(
        **parameter_properties,
        parameter_values=parameter_samples,
        log_likelihoods=log_likelihoods,
    )
    MLE_parameters = calculate_MLE(sample_dataset)

    MLE_output_parameter_dict = change_properties_of_parameters(
        input_parameters,
        change_parameter_values_using_MLE_dataset,
        MLE_parameters,
    )

    output_MLE_filename: Path = Path(
        str(input_parameters_filepath).replace("input", "retrieved")
    )
    with open(output_MLE_filename, "w", newline="") as output_file:
        write_parsed_input_to_output(
            input_file_headers, MLE_output_parameter_dict, output_file
        )

    return None


def make_dataset_from_APOLLO_parameter_file(
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


def get_TP_function_from_APOLLO_parameter_file(
    parameter_filepath: Pathlike, **parsing_kwargs
) -> Callable[[Any], Sequence[float]]:
    with open(parameter_filepath, newline="") as retrieved_file:
        parsed_retrieved_file = parse_APOLLO_input_file(
            retrieved_file, **parsing_kwargs
        )

    TP_function_name = parsed_retrieved_file["parameters"]["Atm"]["options"][0]

    return getattr(TP, TP_function_name)
