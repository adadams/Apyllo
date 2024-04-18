from functools import lru_cache
import numpy as np
from typing import Any, Callable, Sequence

from apollo.Apollo_components import (
    ReadInputsfromFile,
    ProcessInputs,
    MakeObservation,
    MakePlanet,
)
from apollo.general_protocols import Pathlike

OPACITY_DIRECTORY = "/Volumes/ResearchStorage/Opacities_0v10/"


@lru_cache
def prep_inputs_for_model(input_filepath: Pathlike):
    inputs = ReadInputsfromFile(input_filepath)
    inputs["opacdir"] = OPACITY_DIRECTORY

    processed_inputs = ProcessInputs(**inputs)

    planet = MakePlanet(processed_inputs["MakePlanet_kwargs"])

    get_model = planet.MakeModel(processed_inputs["MakeModel_initialization_kwargs"])

    observation = MakeObservation(
        processed_inputs["ModelObservable_initialization_kwargs"]
    )

    binned_wavelengths = processed_inputs[
        "ModelObservable_initialization_kwargs"
    ].data_wavelengths

    return (
        dict(
            model_function=get_model,
            observation=observation,
            model_parameters=processed_inputs["parameters"],
        ),
        binned_wavelengths,
    )


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
