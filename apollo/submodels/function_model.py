import inspect
import sys
from functools import partial
from os.path import abspath
from typing import Any, Callable, Self

import numpy as np
import tomllib
from numpy.typing import NDArray

APOLLO_DIRECTORY = abspath(
    "/Users/arthur/Documents/Astronomy/2019/Retrieval/Code/APOLLO"
)
if APOLLO_DIRECTORY not in sys.path:
    sys.path.append(APOLLO_DIRECTORY)

from apollo.convenience_types import Pathlike  # noqa: E402
from apollo.useful_internal_functions import turn_dictionary_into_string  # noqa: E402

# Protocols are for general practice for implementing something in APOLLO.
# ABCs are for MY implementation of that something.


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
        metadata_print_style: str = "".join(
            [
                f"{argument_name} = {self.loaded_function_arguments[argument_name]} "
                + f"(metadata: {turn_dictionary_into_string(self.metadata.get(argument_name))})\n"
                if argument_name in self.loaded_function_arguments
                else f"{argument_name} "
                + f"(metadata: {turn_dictionary_into_string(self.metadata.get(argument_name))})\n"
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
        return self.__class__(self.function, metadata=self.metadata, **kwargs)


def make_model(
    function: Callable = None,
    path_to_metadata: Pathlike = None,
    metadata_loader: Callable = tomllib.load,
    *function_args,
    **function_kwargs,
):
    if path_to_metadata is not None:
        with open(path_to_metadata, "rb") as metadata_file:
            metadata: dict[str, dict[str, Any]] = metadata_loader(metadata_file)
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
