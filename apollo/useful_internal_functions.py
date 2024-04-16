from collections import ChainMap
from functools import reduce
import numpy as np
from numpy.typing import ArrayLike
from typing import Any, Callable
from xarray import load_dataset
import yaml

from apollo.general_protocols import Pathlike


def compose(*functions):
    return reduce(lambda f, g: lambda x: g(f(x)), functions)


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


def interleave(
    first_terms: ArrayLike, second_terms: ArrayLike, interleaved_axis: int = -1
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
