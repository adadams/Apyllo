import sys
from dataclasses import dataclass
from os.path import abspath
from pathlib import Path
from typing import Any

from xarray import Dataset

APOLLO_DIRECTORY = abspath(
    "/Users/arthur/Documents/Astronomy/2019/Retrieval/Code/APOLLO"
)
sys.path.append(APOLLO_DIRECTORY)

from custom_types import Pathlike  # noqa: E402
from apollo.retrieval.results.build_results_datasets import (  # noqa: E402
    compile_contributions_into_dataset,
    compile_results_into_dataset,
    prepare_MLE_dataset_from_results_dataset,
    prepare_model_spectra_dataset_from_free_parameters_dataset,
    prepare_TP_profile_dataset_from_results_dataset,
)
from apollo.retrieval.results.IO import (  # noqa: E402
    SamplingResults,
    unpack_results_filepaths,
)
from apollo.retrieval.samplers.dynesty.parse_dynesty_outputs import (  # noqa: E402
    load_and_filter_all_parameters_by_importance,
)
from apollo.submodels.TP_models.get_TP_function import (  # noqa: E402
    get_TP_function_from_APOLLO_parameter_file,
)
from user_directories import USER_DIRECTORIES  # noqa: E402

RESULTS_DIRECTORY: Pathlike = USER_DIRECTORIES["results"]


@dataclass
class RetrievalResults:
    parameter_samples: Dataset
    MLE_parameters: Dataset
    contributions: Dataset
    TP_profiles: Dataset
    model_spectra: Dataset


def process_dynesty_run(
    run_filepath_directory: dict[str, Pathlike],
    number_of_resampled_model_spectra: int = 500,
):
    fitting_results_filepath: Pathlike = run_filepath_directory["fitting_results"]
    derived_fit_parameters_filepath: Pathlike = run_filepath_directory[
        "derived_fit_parameters"
    ]
    MLE_parameters_filepath: Pathlike = run_filepath_directory["MLE_parameters"]
    original_contributions_filepath: Pathlike = run_filepath_directory["contributions"]
    output_filepath_plus_prefix: Pathlike = run_filepath_directory["output_file_prefix"]
    file_parsing_kwargs: dict[str, Any] = run_filepath_directory["file_parsing_kwargs"]

    samples_and_likelihoods: SamplingResults = (
        load_and_filter_all_parameters_by_importance(
            fitting_results_filepath, derived_fit_parameters_filepath
        )
    )

    results_dataset = compile_results_into_dataset(
        samples_and_likelihoods,
        MLE_parameters_filepath,
        output_filepath_plus_prefix,
        **file_parsing_kwargs,
    )

    contributions_dataset = compile_contributions_into_dataset(
        original_contributions_filepath,
        MLE_parameters_filepath,
        output_filepath_plus_prefix,
    )

    MLE_dataset = prepare_MLE_dataset_from_results_dataset(
        results_dataset, output_filepath_plus_prefix
    )

    TP_function = get_TP_function_from_APOLLO_parameter_file(
        MLE_parameters_filepath, **file_parsing_kwargs
    )
    TP_profile_dataset = prepare_TP_profile_dataset_from_results_dataset(
        results_dataset, TP_function, output_filepath_plus_prefix
    )

    model_spectra_dataset = prepare_model_spectra_dataset_from_free_parameters_dataset(
        results_dataset,
        MLE_parameters_filepath,
        number_of_resampled_model_spectra,
        output_filepath_plus_prefix,
    )

    return RetrievalResults(
        parameter_samples=results_dataset,
        MLE_parameters=MLE_dataset,
        contributions=contributions_dataset,
        TP_profiles=TP_profile_dataset,
        model_spectra=model_spectra_dataset,
    )


def process_dynesty_runs(
    run_filepath_directories: dict[str, dict[str, Pathlike]],
) -> dict[str, RetrievalResults]:
    return {
        run_name: process_dynesty_run(run_filepath_directory)
        for run_name, run_filepath_directory in run_filepath_directories.items()
    }


if __name__ == "__main__":
    RUN_FOLDER_NAME = "2M2236"

    specific_results_directory: Path = RESULTS_DIRECTORY / RUN_FOLDER_NAME

    run_filepath_directories: dict[str, dict[str, Path]] = unpack_results_filepaths(
        specific_results_directory
    )

    process_dynesty_run(run_filepath_directories["HK+JWST_2M2236_logg-free_cloudy"])
