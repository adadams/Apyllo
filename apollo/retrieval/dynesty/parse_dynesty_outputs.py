import dynesty
import numpy as np
from numpy.typing import ArrayLike
from pathlib import Path
import pickle
from typing import Any, Sequence

from apollo.general_protocols import Pathlike
from apollo.useful_internal_functions import load_multi_yaml_file_into_dict


def unpack_results_filepaths(
    results_directory, directory_yaml_filename: Pathlike = "results_files.yaml"
) -> dict[str, dict[str, Pathlike]]:
    run_results_directories: dict[str, dict[str, str]] = load_multi_yaml_file_into_dict(
        Path(results_directory) / directory_yaml_filename
    )

    # Results directories are all filenames EXCEPT for the parsing keyword arguments,
    # which will be parsed as a dictionary.
    run_filepaths: dict[str, dict[str, Pathlike | dict[str, Any]]] = {
        run_name: {
            entry_name: (
                Path(results_directory) / run_name / entry
                if not isinstance(entry, dict)
                else entry
            )
            for entry_name, entry in run_results_directory.items()
        }
        for run_name, run_results_directory in run_results_directories.items()
    }

    return run_filepaths


def load_dynesty_results(filepath: Pathlike) -> dynesty.results.Results:
    with open(filepath, "rb") as results_file:
        dynesty_results = pickle.load(results_file)

    return dynesty_results


def load_derived_parameters(filepath: Pathlike) -> ArrayLike:
    with open(filepath, "rb") as derived_parameters_file:
        derived_parameters = pickle.load(derived_parameters_file)

    return derived_parameters


def compile_dynesty_parameters(
    dynesty_results: dynesty.results.Results, derived_parameters: ArrayLike
) -> ArrayLike:
    return np.c_[dynesty_results.samples, derived_parameters]


def make_filter_of_dynesty_samples_by_importance(
    dynesty_results: dynesty.results.Results, importance_weight_percentile=0.95
) -> Sequence[bool]:
    importance_weights = dynesty_results.importance_weights()
    cumulative_importance_weights = np.cumsum(importance_weights[::-1])[::-1]

    is_important_enough = cumulative_importance_weights <= importance_weight_percentile
    return is_important_enough


def load_and_filter_all_parameters_by_importance(
    fitting_results_filepath: Pathlike,
    derived_fit_parameters_filepath: Pathlike,
) -> dict[str, ArrayLike]:
    dynesty_results = load_dynesty_results(fitting_results_filepath)

    derived_parameters = load_derived_parameters(derived_fit_parameters_filepath)

    compiled_parameters = compile_dynesty_parameters(
        dynesty_results, derived_parameters
    )

    is_important_enough = make_filter_of_dynesty_samples_by_importance(dynesty_results)

    samples = compiled_parameters[is_important_enough].T
    log_likelihoods = dynesty_results.logl[is_important_enough]

    return dict(samples=samples, log_likelihoods=log_likelihoods)
