from typing import Protocol

from numpy.typing import ArrayLike


class Measured(Protocol):
    lower_errors: ArrayLike
    upper_errors: ArrayLike
    errors: ArrayLike
