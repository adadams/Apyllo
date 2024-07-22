import pickle
import sys
from os.path import abspath
from pathlib import Path

import numpy as np
from deepdiff import DeepDiff
from numpy.typing import NDArray
from xarray import Dataset, load_dataset

APOLLO_DIRECTORY = abspath(
    "/Users/arthur/Documents/Astronomy/2019/Retrieval/Code/APOLLO"
)
sys.path.append(APOLLO_DIRECTORY)

from apollo.Apollo_components import ReadInputsfromFile  # noqa: E402
from custom_types import Pathlike  # noqa: E402
from user_directories import USER_DIRECTORIES  # noqa: E402

RESULTS_DIRECTORY: Pathlike = USER_DIRECTORIES["results"]


results_directory: Path = Path.cwd() / RESULTS_DIRECTORY


def prep_inputs_for_model(input_filepath: Pathlike) -> dict:
    inputs: dict = ReadInputsfromFile(input_filepath)

    with open("test_input_pickle.pkl", "wb") as test_file:
        pickle.dump(inputs, test_file)


def check_inputs_are_preserved(
    input_filepath: Pathlike, pickle_input_filepath: Pathlike
):
    inputs: dict = ReadInputsfromFile(input_filepath)

    with open(pickle_input_filepath, "rb") as test_file:
        pickle_inputs: dict = pickle.load(test_file)

    assert not DeepDiff(inputs, pickle_inputs)


def quick_replace_input_parameters(
    inputs: dict, MLE_parameter_values: NDArray[np.float_]
) -> dict:
    MLE_dictionary = inputs.copy()

    MLE_dictionary["plparams"] = MLE_parameter_values
    MLE_dictionary["guess"] = MLE_parameter_values

    return MLE_dictionary


test_results_directory: Path = (
    results_directory / "2M2236" / "HK+JWST_2M2236_logg-free_cloudy"
)

test_input_filepath: Path = (
    test_results_directory
    / "2M2236.Piette.HK+G395H.cloudy.input.2024-05-21.resume-from-checkpoint.dat"
)

test_MLE_dataset: Dataset = load_dataset(
    test_results_directory / "2M2236.Piette.HK+G395H.cloudy.MLE-parameters.nc"
)

# prep_inputs_for_model(test_input_filepath)
with open("test_input_pickle.pkl", "rb") as test_file:
    test_inputs: dict = pickle.load(test_file)


test_MLE = quick_replace_input_parameters(
    test_inputs, test_MLE_dataset.to_array().values
)

with open("test_MLE_pickle.pkl", "wb") as test_file:
    pickle.dump(test_MLE, test_file)
