from collections import ChainMap
from collections.abc import Callable
from functools import reduce
from inspect import Signature
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from apollo.formats.custom_types import Pathlike


def compose(*functions):
    return reduce(lambda f, g: lambda x: g(f(x)), functions)


def count_number_of_arguments(function: Callable) -> int:
    return len(Signature.from_callable(function).parameters)


def load_multi_yaml_file_into_dict(filepath: Pathlike):
    with open(filepath, "r") as multi_file:
        loaded_file = yaml.safe_load_all(multi_file)

        file_dict = ChainMap(*loaded_file)

    return file_dict


def format_yaml_from_template(
    input_name: str, template: str, formatter: Callable
) -> dict[str, dict[str, Any]]:
    formats = formatter(input_name)

    formatted_template = template
    for template_name, specific_name in formats.items():
        formatted_template = formatted_template.replace(template_name, specific_name)

    return yaml.safe_load(formatted_template)


def turn_dictionary_into_string(dictionary: dict[str, Any]) -> str:
    return (
        str(dictionary)
        .replace("{", "")
        .replace("}", "")
        .replace(":", " =")
        .replace("'", "")
    )


def interleave(
    first_terms: NDArray[np.float_],
    second_terms: NDArray[np.float_],
    interleaved_axis: int = -1,
):
    interleaved_dimension_size = (
        np.shape(first_terms)[interleaved_axis]
        + np.shape(second_terms)[interleaved_axis]
    )
    interleaved_array_shape = np.shape(first_terms)
    interleaved_array_shape[interleaved_axis] = interleaved_dimension_size
    interleaved_array = np.empty(interleaved_array_shape, dtype=first_terms.dtype)

    base_slice_list = [slice(None)] * first_terms.ndim

    first_slice = slice(0, None, 2)
    first_slices = base_slice_list
    first_slices[interleaved_axis] = first_slice

    second_slice = slice(1, None, 2)
    second_slices = base_slice_list
    second_slices[interleaved_axis] = second_slice

    interleaved_array[first_slices] = first_terms
    interleaved_array[second_slices] = second_terms

    return interleaved_array


def strtobool(value: str) -> bool:
    value = value.lower()
    if value in ("y", "yes", "on", "1", "true", "t"):
        return True
    return False


def get_from_dictionary(input_dictionary: dict[str, Any], item_name: str) -> list[Any]:
    return [entry[item_name] for entry in input_dictionary.values()]


def get_across_dictionaries(
    input_dictionary: dict[str, dict[str, Any]], item_name: str
) -> dict[str, list[Any]]:
    return {
        name: get_from_dictionary(nested_dictionary, item_name)
        for name, nested_dictionary in input_dictionary.items()
    }
