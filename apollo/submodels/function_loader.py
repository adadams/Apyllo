import inspect
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Callable, Sequence

from numpy.typing import ArrayLike

# Protocols are for general practice for implementing something in APOLLO.
# ABCs are for MY implementation of that something.


# Is this even an ABC, or just a base class? We will need to try this implementation out
# and see what can be abstracted and what can't.
class FunctionLoader(ABC):
    def __init__(self, function: Callable, *function_args, **function_kwargs) -> None:
        self.function = function
        self._loaded_function = partial(
            self.function, *function_args, **function_kwargs
        )
        self._loaded_arguments: dict[str, Any] = (
            inspect.signature(self.function)
            .bind_partial(*function_args, **function_kwargs)
            .arguments
        )

    @abstractmethod
    def __repr__(self) -> str: ...

    @abstractmethod
    def __call__(self, pressures: float | Sequence[float]) -> ArrayLike: ...
