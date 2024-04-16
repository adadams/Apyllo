##########################################
# BOILERPLATE CODE TO RESOLVE APOLLO PATH
from os.path import abspath
import sys

APOLLO_DIRECTORY = abspath(
    "/Users/arthur/Documents/Astronomy/2019/Retrieval/Code/APOLLO"
)
sys.path.append(APOLLO_DIRECTORY)
##########################################

from pathlib import Path
from typing import Any
from xarray import Dataset

from apollo.general_protocols import Pathlike
from apollo.retrieval.dynesty.parse_dynesty_outputs import unpack_results_filepaths
from apollo.retrieval.dynesty.apollo_interface_functions import (
    get_TP_function_from_APOLLO_parameter_file,
)
from apollo.retrieval.dynesty.prepare_final_results_datasets import (
    compile_results_into_dataset,
    compile_contributions_into_dataset,
    prepare_MLE_dataset_from_results_dataset,
    prepare_TP_profile_dataset_from_results_dataset,
    prepare_model_spectra_dataset_from_free_parameters_dataset,
)

from user_directories import USER_DIRECTORIES


def main(
    run_filepath_directories: dict[str, dict[str, Pathlike]],
    dataset_save_directory: Pathlike,
) -> dict[str, dict[str, Dataset]]:
    processed_datasets = {}

    for run_name, run_filepath_directory in run_filepath_directories.items():
        fitting_results_filepath: Pathlike = run_filepath_directory["fitting_results"]

        derived_fit_parameters_filepath: Pathlike = run_filepath_directory[
            "derived_fit_parameters"
        ]

        MLE_parameters_filepath: Pathlike = run_filepath_directory["MLE_parameters"]

        original_contributions_filepath: Pathlike = run_filepath_directory[
            "contributions"
        ]

        output_filepath_plus_prefix: Pathlike = (
            Path(dataset_save_directory)
            / run_name
            / run_filepath_directory["output_file_prefix"]
        )

        file_parsing_kwargs: dict[str, Any] = run_filepath_directory[
            "file_parsing_kwargs"
        ]

        results_dataset = compile_results_into_dataset(
            fitting_results_filepath,
            derived_fit_parameters_filepath,
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

        model_spectra_dataset = (
            prepare_model_spectra_dataset_from_free_parameters_dataset(
                results_dataset,
                MLE_parameters_filepath,
                output_filepath_plus_prefix,
                number_of_resampled_model_spectra=500,
            )
        )

        processed_datasets[run_name] = dict(
            samples=results_dataset,
            MLE=MLE_dataset,
            contributions=contributions_dataset,
            TP_profiles=TP_profile_dataset,
            model_spectra=model_spectra_dataset,
        )

    return processed_datasets


RESULTS_DIRECTORY: Pathlike = USER_DIRECTORIES["results"]
RESULTS_DIRECTORY_2M2236: Pathlike = RESULTS_DIRECTORY / "2M2236"


if __name__ == "__main__":
    run_filepath_directories: dict[str, dict[str, Pathlike]] = unpack_results_filepaths(
        RESULTS_DIRECTORY_2M2236
    )

    main(run_filepath_directories, dataset_save_directory=RESULTS_DIRECTORY_2M2236)
