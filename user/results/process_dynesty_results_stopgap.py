import sys
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray
from pandas import read_pickle
from xarray import Dataset

APOLLO_DIRECTORY = Path.home() / "Documents" / "Astronomy" / "code" / "Apyllo"
sys.path.append(str(APOLLO_DIRECTORY))

from apollo.retrieval.results.build_results_datasets import (  # noqa: E402
    compile_contributions_into_dataset,
    compile_results_into_dataset,
    prepare_MLE_dataset_from_results_dataset,
    prepare_model_spectra_dataset_from_free_parameters_dataset,
    prepare_TP_profile_dataset_from_results_dataset,
)
from apollo.retrieval.results.IO import (  # noqa: E402
    SamplingResults,
    make_MLE_parameter_file_from_input_parameter_file,
    unpack_results_filepaths,
)
from apollo.retrieval.samplers.dynesty.parse_dynesty_outputs import (  # noqa: E402
    load_and_filter_all_parameters_by_importance,
)
from apollo.submodels.TP_models.get_TP_function import (  # noqa: E402
    get_TP_function_from_APOLLO_parameter_file,
)
from custom_types import Pathlike  # noqa: E402
from user_directories import USER_DIRECTORIES  # noqa: E402

RESULTS_DIRECTORY: Pathlike = USER_DIRECTORIES["results"]


class RetrievalResults(NamedTuple):
    parameter_samples: Dataset
    MLE_parameters: Dataset
    contributions: Dataset
    TP_profiles: Dataset
    model_spectra: Dataset


def process_dynesty_run(
    run_filepath_directory: dict[str, Pathlike],
    number_of_resampled_model_spectra: int = 500,
) -> RetrievalResults:
    fitting_results_filepath: Pathlike = run_filepath_directory["fitting_results"]
    derived_fit_parameters_filepath: Pathlike = run_filepath_directory[
        "derived_fit_parameters"
    ]
    data_filepath: Pathlike = run_filepath_directory["data"]
    input_parameters_filepath: Pathlike = run_filepath_directory["input_parameters"]
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
        input_parameters_filepath,
        output_filepath_plus_prefix,
        **file_parsing_kwargs,
    )

    print(f"{results_dataset=}")

    data_array = np.loadtxt(data_filepath)
    data_wavelengths = (data_array[:, 0] + data_array[:, 1]) / 2
    # print(f"{data_wavelengths=}")

    MLE_dataset = prepare_MLE_dataset_from_results_dataset(
        results_dataset, output_filepath_plus_prefix
    )
    # print(f"{MLE_dataset=}")

    make_MLE_parameter_file_from_input_parameter_file(
        MLE_parameters=MLE_dataset,
        input_parameters_filepath=input_parameters_filepath,
        data_filepath=data_filepath,
    )

    with open(original_contributions_filepath, "rb") as pickle_file:
        original_contributions: dict[str, NDArray[np.float64]] = read_pickle(
            pickle_file
        )

    contributions_dataset = compile_contributions_into_dataset(
        contributions=original_contributions,
        binned_wavelengths=data_wavelengths,
        output_filepath_plus_prefix=output_filepath_plus_prefix,
    )

    retrieved_file_parsing_kwargs = {
        key: value for key, value in file_parsing_kwargs.items()
    }
    retrieved_file_parsing_kwargs["delimiter"] = "\t"
    # if this code generates the retrieved parameter file,
    # regardless of how the original file was delimited, it will be a tab-delimited file

    TP_function = get_TP_function_from_APOLLO_parameter_file(
        MLE_parameters_filepath, **retrieved_file_parsing_kwargs
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

    process_dynesty_run(
        run_filepath_directories["JWST-only_2M2236_logg-free/self-retrieval_test"]
    )
