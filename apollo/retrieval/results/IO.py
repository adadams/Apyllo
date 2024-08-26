import sys
from collections.abc import Sequence
from dataclasses import dataclass
from operator import itemgetter
from os.path import abspath
from pathlib import Path
from typing import Any

import numpy as np
import tomllib
from numpy.typing import NDArray
from xarray import Dataset

from apollo.retrieval.results.manipulate_results_datasets import (  # noqa: E402
    change_parameter_values_using_MLE_dataset,
)
from custom_types import Pathlike  # noqa: E402
from useful_internal_functions import load_multi_yaml_file_into_dict
from user.forward_models.inputs.parse_APOLLO_inputs import (  # noqa: E402
    change_properties_of_parameters,
    parse_APOLLO_input_file,
    write_parsed_input_to_output,
)


@dataclass
class SamplingResults:
    samples: NDArray[np.float_]
    log_likelihoods: NDArray[np.float_]


def unpack_results_filepaths(
    results_directory: Pathlike,
    directory_yaml_filename: Pathlike = "results_files.yaml",
) -> dict[str, dict[str, Pathlike]]:
    run_results_directories: dict[str, dict[str, str]] = load_multi_yaml_file_into_dict(
        Path(results_directory) / directory_yaml_filename
    )

    # Results directories are all filenames EXCEPT for the parsing keyword arguments,
    # which will be parsed as a dictionary.
    run_filepaths: dict[str, dict[str, Pathlike | dict[str, Any]]] = {
        run_name: {
            entry_name: (
                # This assumes you put all the needed files in the results folder for the run.
                # This could be generalized if the results_files.yaml had relative paths to begin with.
                Path(results_directory) / run_name / entry
                if not isinstance(entry, dict)
                else entry
            )
            for entry_name, entry in run_results_directory.items()
        }
        for run_name, run_results_directory in run_results_directories.items()
    }

    return run_filepaths


APOLLO_DIRECTORY = abspath(
    "/Users/arthur/Documents/Astronomy/2019/Retrieval/Code/APOLLO"
)
if APOLLO_DIRECTORY not in sys.path:
    sys.path.append(APOLLO_DIRECTORY)


def guess_default_units_from_parameter_names(
    parameter_names: Sequence[str],
) -> list[str]:
    guessed_units = []

    for parameter_name in parameter_names:
        if "rad" in parameter_name.lower():
            guessed_unit = "Earth_radii"

        elif "mass" in parameter_name.lower():
            guessed_unit = "Jupiter_masses"

        elif ("T_" in parameter_name) or (parameter_name == "Teff"):
            guessed_unit = "kelvin"

        elif parameter_name == "deltaL":
            guessed_unit = "nanometers"

        else:
            guessed_unit = "dimensionless"

        guessed_units.append(guessed_unit)

    return guessed_units


def guess_default_string_formats_from_parameter_names(
    parameter_names: Sequence[str],
) -> list[str]:
    guessed_float_precisions = []

    for parameter_name in parameter_names:
        if "mass" in parameter_name.lower():
            guessed_float_precision = ".0f"

        elif ("T_" in parameter_name) or (parameter_name == "Teff"):
            guessed_float_precision = ".0f"

        else:
            guessed_float_precision = ".2f"

        guessed_float_precisions.append(guessed_float_precision)

    return guessed_float_precisions


def get_parameter_properties_from_defaults(
    free_parameter_names: Sequence[str],
    free_parameter_group_slices: Sequence[slice],
    derived_parameter_names: Sequence[str] = [
        "Mass",
        "C/O",
        "[Fe/H]",
        "Teff",
    ],
) -> dict[str, Any]:
    parameter_names = free_parameter_names + derived_parameter_names

    parameter_units = guess_default_units_from_parameter_names(parameter_names)

    free_parameter_group_names = np.empty_like(free_parameter_names)
    for group_name, group_slice in free_parameter_group_slices.items():
        free_parameter_group_names[group_slice] = group_name

    derived_parameter_group_names = ["Derived"] * len(derived_parameter_names)

    parameter_group_names = (
        list(free_parameter_group_names) + derived_parameter_group_names
    )

    parameter_default_string_formattings = (
        guess_default_string_formats_from_parameter_names(parameter_names)
    )

    return {
        "parameter_names": parameter_names,
        "parameter_units": parameter_units,
        "parameter_default_string_formattings": parameter_default_string_formattings,
        "parameter_group_names": parameter_group_names,
    }


def get_print_names_from_parameter_names(
    parameter_names: Sequence[str], parameter_spec_filepath: Pathlike
) -> Sequence[str]:
    with open(parameter_spec_filepath, "rb") as parameter_spec_file:
        parameter_specs = tomllib.load(parameter_spec_file)

    name_getter = itemgetter(*parameter_names)

    return [
        parameter_spec["print_name"] for parameter_spec in name_getter(parameter_specs)
    ]


def create_MLE_output_dictionary(
    MLE_parameters: Dataset,
    input_parameters_filepath: Pathlike,
) -> dict:
    # parameter_samples = results.samples
    # log_likelihoods = results.log_likelihoods

    with open(input_parameters_filepath, newline="") as input_file:
        parsed_input_file = parse_APOLLO_input_file(input_file, delimiter=" ")

    input_parameters = parsed_input_file["parameters"]
    # input_parameter_names = parsed_input_file["parameter_names"]
    # input_parameter_group_slices = parsed_input_file["parameter_group_slices"]

    # parameter_properties = get_parameter_properties_from_defaults(
    #    input_parameter_names, input_parameter_group_slices
    # )
    # parameter_print_names = get_print_names_from_parameter_names(
    #    input_parameter_names, "user/results/2M2236/parameter_print_attributes.toml"
    # )

    # sample_dataset = make_run_parameter_dataset(
    #    **parameter_properties,
    #    parameter_print_names=parameter_print_names,
    #    parameter_values=parameter_samples,
    #    log_likelihoods=log_likelihoods,
    # )
    # MLE_parameters = calculate_MLE(sample_dataset)

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
    # results: SamplingResults,
    MLE_parameters: Dataset,
    input_parameters_filepath: Pathlike,
    data_filepath: Pathlike,
) -> None:
    # parameter_samples = results.samples
    # log_likelihoods = results.log_likelihoods

    with open(input_parameters_filepath, newline="") as input_file:
        parsed_input_file = parse_APOLLO_input_file(input_file, delimiter=" ")

    input_parameters = parsed_input_file["parameters"]
    # input_parameter_names = parsed_input_file["parameter_names"]
    # input_parameter_group_slices = parsed_input_file["parameter_group_slices"]
    input_file_headers = parsed_input_file["header"]

    input_file_headers["Data"] = (str(data_filepath.relative_to(Path.cwd())),)

    # parameter_properties = get_parameter_properties_from_defaults(
    #    input_parameter_names, input_parameter_group_slices
    # )

    # sample_dataset = make_run_parameter_dataset(
    #    **parameter_properties,
    #    parameter_values=parameter_samples,
    #    log_likelihoods=log_likelihoods,
    # )
    # MLE_parameters = calculate_MLE(sample_dataset)

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
