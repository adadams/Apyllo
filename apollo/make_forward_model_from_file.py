from functools import lru_cache
from typing import Any, Callable, Sequence, TypedDict

import numpy as np
from numpy.typing import NDArray

from apollo.Apollo_components import (
    MakeObservation,
    MakePlanet,
    ProcessedInputs,
    ProcessInputs,
    ReadInputsfromFile,
)
from apollo.formats.custom_types import Pathlike
from apollo.planet import Planet

OPACITY_DIRECTORY = "/Volumes/ResearchStorage/Opacities_0v10/"


class PreppedInputs(TypedDict):
    model_function: Callable[[Any], Any]
    observation: Callable[[Any], Any]
    model_parameters: Sequence[float]


@lru_cache
def prep_inputs_for_model(input_filepath: Pathlike) -> dict:
    inputs: dict = ReadInputsfromFile(input_filepath)
    inputs["opacdir"] = OPACITY_DIRECTORY

    processed_inputs: ProcessedInputs = ProcessInputs(**inputs)

    planet: Planet = MakePlanet(processed_inputs["MakePlanet_kwargs"])

    get_model: Callable = planet.MakeModel(
        processed_inputs["MakeModel_initialization_kwargs"]
    )

    observation: Callable = MakeObservation(
        processed_inputs["ModelObservable_initialization_kwargs"]
    )

    binned_wavelengths: NDArray[np.float_] = processed_inputs[
        "ModelObservable_initialization_kwargs"
    ]["data_wavelengths"]

    return {
        "prepped_inputs": PreppedInputs(
            model_function=get_model,
            observation=observation,
            model_parameters=processed_inputs["parameters"],
        ),
        "binned_wavelengths": binned_wavelengths,
    }


def evaluate_model_spectrum(
    model_function: Callable[[Any], Any],
    observation: Callable[[Any], Any],
    model_parameters: Sequence[float],
):
    model = model_function(model_parameters)

    obs_args = [
        np.asarray(model[0]),  # model spectrum in emission flux
        model_parameters[0],  # radius
        "Rad",  # using radius and not R/D or some derivative thereof
        model_parameters[-4],  # delta lambda index
    ]

    observed_binned_model, observed_full_resolution_model = observation(*obs_args)

    return observed_binned_model["data"]


def generate_model_spectrum(input_filepath):
    prepped_inputs, wavelengths = prep_inputs_for_model(input_filepath)

    observed_binned_model_spectrum = evaluate_model_spectrum(**prepped_inputs)
    return observed_binned_model_spectrum
