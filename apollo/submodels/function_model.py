import inspect
from functools import partial
from typing import Any, Callable, Self

import msgspec
import numpy as np
from numpy.typing import NDArray

from custom_types import Pathlike  # noqa: E402
from useful_internal_functions import turn_dictionary_into_string  # noqa: E402


# NOTE: could do a class decorator instead of function decorator?
# How should we treat the current construction where we can sub-sequently load
# parameter values and return the same object?
# I think this is okay because the class can initially be constructed without loaded arguments.
class FunctionModel:
    def __init__(
        self,
        function: Callable,
        metadata: dict[str, dict[str, Any]] = None,
        *function_args,
        **function_kwargs,
    ) -> Self:
        self.function: Callable = function
        self.function_arguments: list[str] = inspect.getfullargspec(self.function).args

        self.free_arguments: list[str] = [
            function_argument
            for function_argument in self.function_arguments
            if function_argument not in metadata["dimensional_arguments"]
        ]

        self.loaded_function: Callable = (
            partial(function, *function_args, **function_kwargs)
            if function_kwargs
            else function
        )
        self.loaded_function_arguments: dict[str, Any] = (
            (
                inspect.signature(self.function)
                .bind_partial(*function_args, **function_kwargs)
                .arguments
            )
            if function_kwargs
            else dict()
        )

        self.metadata: dict[str, dict[str, Any]] = metadata if metadata else dict()

    def __repr__(self) -> str:
        parameter_metadata = self.metadata["parameters"]

        metadata_print_style: str = "".join(
            [
                f"{argument_name} = {self.loaded_function_arguments[argument_name]} "
                + f"(metadata: {turn_dictionary_into_string(parameter_metadata.get(argument_name))})\n"
                if argument_name in self.loaded_function_arguments
                else f"{argument_name} "
                + f"(metadata: {turn_dictionary_into_string(parameter_metadata.get(argument_name))})\n"
                for argument_name in self.function_arguments
            ]
        )
        return (
            f"Function: {self.function.__name__} \n"
            + "Arguments: \n"
            + f"{metadata_print_style}"
        )

    def __call__(self, *args, **kwargs) -> NDArray[np.float_]:
        return self.loaded_function(*args, **kwargs)

    def load_parameters(self, *args, **kwargs) -> Self:
        return self.__class__(self.function, *args, metadata=self.metadata, **kwargs)


def make_model(
    function: Callable = None,
    path_to_metadata: Pathlike = None,
    metadata_loader: Callable = msgspec.toml.decode,
    *function_args,
    **function_kwargs,
):
    if path_to_metadata is not None:
        metadata: dict[str, dict[str, Any]] = metadata_loader(path_to_metadata)
    else:
        function_argument_names: list[str] = inspect.getfullargspec(function).args
        metadata: dict[str, None] = dict.fromkeys(function_argument_names, None)

    if function:
        return FunctionModel(
            function=function, metadata=metadata, *function_args, **function_kwargs
        )
    else:

        def construct_FunctionModel(function):
            return FunctionModel(
                function=function, metadata=metadata, *function_args, **function_kwargs
            )

        return construct_FunctionModel


# FUTURE: this might be a good construction for function *templates*, rather than
# function *instances*. For example, a cloud model that implements different cloud
# species, or a modified version of the Piette profile, or each gas species implementing
# Rayleigh scattering and its own opacity table read. Is this just reimplementing something
# obvious?

# FUTURE: store the signature in the class, and use properties to access various parts of it?
