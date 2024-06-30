from collections.abc import Callable, Sequence
from typing import Any

from apollo.formats.custom_types import Pathlike
from apollo.submodels import TP
from user.forward_models.inputs.parse_APOLLO_inputs import parse_APOLLO_input_file


def get_TP_function_from_APOLLO_parameter_file(
    parameter_filepath: Pathlike, **parsing_kwargs
) -> Callable[[Any], Sequence[float]]:  # noqa: F821
    with open(parameter_filepath, newline="") as retrieved_file:
        parsed_retrieved_file = parse_APOLLO_input_file(
            retrieved_file, **parsing_kwargs
        )

    TP_function_name = parsed_retrieved_file["parameters"]["Atm"]["options"][0]

    return getattr(TP, TP_function_name)
